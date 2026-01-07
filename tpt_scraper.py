"""
TPT Scraper - Refactored Version
Three main workflows: Search, Scrape Metadata, Download Free Files
With support for multiple named configurations
"""
import asyncio
import aiohttp
import aiohttp_client_cache
import async_timeout
import random
import json
import logging
import aiosqlite
import os
import re
import argparse
import time
from pathlib import Path
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from tqdm.asyncio import tqdm_asyncio
from collections import deque
from typing import Optional, Dict, Any, List, Tuple
from config_manager import ConfigManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

###########################
# Database Setup          #
###########################

async def setup_db(db_file: str):
    """Creates the SQLite database and tables if they don't exist."""
    async with aiosqlite.connect(db_file) as db:
        # Performance settings
        await db.execute("PRAGMA journal_mode=WAL;")
        await db.execute("PRAGMA synchronous=NORMAL;")
        await db.execute("PRAGMA cache_size=10000;")
        await db.execute("PRAGMA temp_store=MEMORY;")
        
        # URLs discovered during search
        await db.execute("""
            CREATE TABLE IF NOT EXISTS search_results (
                url TEXT PRIMARY KEY,
                resource_type TEXT,
                grade_level TEXT,
                subject TEXT,
                format TEXT,
                price_option TEXT,
                supports TEXT,
                sort_order TEXT,
                page INTEGER,
                discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Product metadata
        await db.execute("""
            CREATE TABLE IF NOT EXISTS product_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE,
                title TEXT,
                short_description TEXT,
                long_description TEXT,
                rating_value TEXT,
                number_of_ratings TEXT,
                product_price TEXT,
                preview_keywords TEXT,
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Downloaded files
        await db.execute("""
            CREATE TABLE IF NOT EXISTS downloads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_url TEXT UNIQUE,
                file_path TEXT,
                file_size INTEGER,
                downloaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Download queue - manually populate this table to queue specific products for download
        await db.execute("""
            CREATE TABLE IF NOT EXISTS download_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_url TEXT UNIQUE,
                priority INTEGER DEFAULT 0,
                notes TEXT,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        await db.execute("CREATE INDEX IF NOT EXISTS idx_search_price ON search_results(price_option);")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_search_resource ON search_results(resource_type);")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_metadata_url ON product_metadata(url);")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_queue_priority ON download_queue(priority DESC);")
        
        await db.commit()

###########################
# Adaptive Components     #
###########################

class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts based on response times and errors."""
    
    def __init__(self, initial_delay=1.0, max_delay=30.0, error_threshold=0.1):
        self.current_delay = initial_delay
        self.max_delay = max_delay
        self.error_threshold = error_threshold
        self.recent_responses = deque(maxlen=100)
        self.error_count = 0
        self.total_count = 0
        
    def record_response(self, success: bool, response_time: float):
        """Record a response and its timing."""
        self.recent_responses.append((success, response_time, time.time()))
        self.total_count += 1
        if not success:
            self.error_count += 1
            
    def adjust_delay(self):
        """Adjust delay based on recent performance."""
        if len(self.recent_responses) < 10:
            return self.current_delay
            
        recent_errors = sum(1 for success, _, _ in self.recent_responses if not success)
        error_rate = recent_errors / len(self.recent_responses)
        avg_response_time = sum(rt for _, rt, _ in self.recent_responses) / len(self.recent_responses)
        
        if error_rate > self.error_threshold:
            self.current_delay = min(self.current_delay * 1.5, self.max_delay)
        elif error_rate < self.error_threshold / 2 and avg_response_time < 2.0:
            self.current_delay = max(self.current_delay * 0.9, 0.1)
            
        return self.current_delay
        
    async def wait(self):
        """Wait for the current delay period."""
        delay = self.adjust_delay()
        await asyncio.sleep(delay)

###########################
# URL Building and Fetch  #
###########################

def build_page_url(resource_type, grade_level, subject, format_type, price_option, supports, sort_order, page):
    """Build a TPT browse URL."""
    url_parts = ["https://www.teacherspayteachers.com/browse"]
    
    if resource_type: url_parts.append(resource_type)
    if grade_level: url_parts.append(grade_level)
    if subject: url_parts.append(subject)
    if format_type: url_parts.append(format_type)
    if price_option: url_parts.append(price_option)
    if supports: url_parts.append(supports)
    
    base_url = "/".join(url_parts)
    query_params = []
    
    if sort_order and sort_order != "Relevance":
        query_params.append(f"order={sort_order}")
    if page > 1:
        query_params.append(f"page={page}")
    
    return f"{base_url}?{'&'.join(query_params)}" if query_params else base_url

async def fetch(session, url, rate_limiter: Optional[AdaptiveRateLimiter] = None, max_retries=3):
    """Fetch a URL with retry logic."""
    retries = 0
    backoff = 1
    
    while retries < max_retries:
        start_time = time.time()
        try:
            if rate_limiter:
                await rate_limiter.wait()
                
            async with async_timeout.timeout(30):
                async with session.get(url) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        if rate_limiter:
                            rate_limiter.record_response(True, response_time)
                        return await response.text()
                    elif response.status == 429:
                        if rate_limiter:
                            rate_limiter.record_response(False, response_time)
                        await asyncio.sleep(backoff)
                        raise Exception(f"Rate limited (429)")
                    elif response.status >= 500:
                        if rate_limiter:
                            rate_limiter.record_response(False, response_time)
                        raise Exception(f"Server error {response.status}")
                    else:
                        if rate_limiter:
                            rate_limiter.record_response(False, response_time)
                        return None
                        
        except Exception as e:
            retries += 1
            if retries < max_retries:
                await asyncio.sleep(backoff)
                backoff *= 2
                
    logging.error(f"Failed to fetch {url} after {max_retries} retries")
    return None

###########################
# 1. SEARCH WORKFLOW      #
###########################

async def search_urls(config_name: str, config: Dict[str, Any], db_file: str):
    """
    Search for product URLs based on configuration.
    Discovers all product URLs matching the search criteria.
    """
    logging.info(f"=" * 60)
    logging.info(f"SEARCH WORKFLOW - Configuration: {config_name}")
    logging.info(f"=" * 60)
    
    # Extract configuration
    resource_types = config.get("resource_type", [""])
    grade_levels = config.get("grade_level", [""])
    subjects = config.get("subject", [""])
    formats = config.get("format", [""])
    price_options = config.get("price_options", [""])
    supports = config.get("supports", [""])
    sorting_methods = config.get("sorting_methods", ["Relevance"])
    total_pages = config.get("total_pages", 42)
    concurrent_requests = config.get("concurrent_requests", 25)
    
    # Generate all combinations
    combinations = [
        (resource_type, grade_level, subject, format_type, price_option, supports_val, sort_order, page)
        for resource_type in resource_types
        for grade_level in grade_levels
        for subject in subjects
        for format_type in formats
        for price_option in price_options
        for supports_val in supports
        for sort_order in sorting_methods
        for page in range(1, total_pages + 1)
    ]
    
    # Check what's already been searched
    async with aiosqlite.connect(db_file) as db:
        # Create a temp table to track searched combinations
        await db.execute("""
            CREATE TABLE IF NOT EXISTS searched_combinations (
                resource_type TEXT,
                grade_level TEXT,
                subject TEXT,
                format TEXT,
                price_option TEXT,
                supports TEXT,
                sort_order TEXT,
                page INTEGER,
                PRIMARY KEY(resource_type, grade_level, subject, format, price_option, supports, sort_order, page)
            )
        """)
        
        async with db.execute("SELECT * FROM searched_combinations") as cursor:
            searched = set(await cursor.fetchall())
    
    remaining = [c for c in combinations if c not in searched]
    
    logging.info(f"Total combinations: {len(combinations):,}")
    logging.info(f"Already searched: {len(searched):,}")
    logging.info(f"Remaining: {len(remaining):,}")
    
    if not remaining:
        logging.info("All combinations already searched!")
        return
    
    # Initialize components
    rate_limiter = AdaptiveRateLimiter(initial_delay=0.5, max_delay=30.0)
    semaphore = asyncio.Semaphore(concurrent_requests)
    
    connector = aiohttp.TCPConnector(
        limit=concurrent_requests + 10,
        limit_per_host=concurrent_requests,
        keepalive_timeout=30
    )
    timeout = aiohttp.ClientTimeout(total=60, connect=10)
    
    async def search_page(combo):
        """Search a single page and store results."""
        resource_type, grade_level, subject, format_type, price_option, supports_val, sort_order, page = combo
        
        async with semaphore:
            url = build_page_url(resource_type, grade_level, subject, format_type, price_option, supports_val, sort_order, page)
            logging.info(f"Searching: {url}")
            
            html = await fetch(session, url, rate_limiter)
            if not html:
                return []
            
            soup = BeautifulSoup(html, "lxml")
            product_elements = soup.select("a.ProductRowCard-module__cardTitleLink--YPqiC")
            
            urls_found = []
            for product in product_elements:
                href = product.get("href")
                if href:
                    full_url = f"https://www.teacherspayteachers.com{href}" if not href.startswith("http") else href
                    urls_found.append((
                        full_url, resource_type, grade_level, subject, format_type, 
                        price_option, supports_val, sort_order, page
                    ))
            
            # Store results
            async with aiosqlite.connect(db_file) as db:
                # Store URLs
                for url_data in urls_found:
                    await db.execute("""
                        INSERT OR IGNORE INTO search_results 
                        (url, resource_type, grade_level, subject, format, price_option, supports, sort_order, page)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, url_data)
                
                # Mark combination as searched
                await db.execute("""
                    INSERT OR IGNORE INTO searched_combinations 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, combo)
                
                await db.commit()
            
            return urls_found
    
    async with aiohttp_client_cache.CachedSession(
        cache_name="aiohttp_cache", 
        expire_after=3600,
        connector=connector,
        timeout=timeout
    ) as session:
        
        # Process in batches
        batch_size = 100
        for i in range(0, len(remaining), batch_size):
            batch = remaining[i:i+batch_size]
            logging.info(f"Processing batch {i//batch_size + 1}/{(len(remaining) + batch_size - 1)//batch_size}")
            
            tasks = [search_page(combo) for combo in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            total_urls = sum(len(r) for r in results if not isinstance(r, Exception))
            logging.info(f"Found {total_urls} URLs in this batch")
            
            # Sleep between batches
            if i + batch_size < len(remaining):
                await asyncio.sleep(random.uniform(1, 3))
    
    # Final stats
    async with aiosqlite.connect(db_file) as db:
        async with db.execute("SELECT COUNT(DISTINCT url) FROM search_results") as cursor:
            total_urls = (await cursor.fetchone())[0]
    
    logging.info(f"=" * 60)
    logging.info(f"SEARCH COMPLETE")
    logging.info(f"Total unique URLs found: {total_urls:,}")
    logging.info(f"=" * 60)

###########################
# 2. SCRAPE METADATA      #
###########################

def extract_text_with_spacing(element):
    """Extract text from BeautifulSoup element while preserving spacing."""
    if not element:
        return None
    text = element.get_text(separator=' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text if text else None

async def scrape_product_metadata(session, url, rate_limiter: Optional[AdaptiveRateLimiter] = None):
    """Scrape metadata from a single product URL."""
    html = await fetch(session, url, rate_limiter)
    if not html:
        return None
    
    soup = BeautifulSoup(html, "lxml")
    
    # Extract metadata
    title = soup.title.string if soup.title else None
    
    meta_desc = soup.find("meta", {"name": "description"})
    short_description = meta_desc["content"] if meta_desc and meta_desc.has_attr("content") else None
    
    long_desc_elem = soup.select_one('div[class*="htmlDisplay"]')
    long_description = extract_text_with_spacing(long_desc_elem) if long_desc_elem else None
    
    rating_value = None
    number_of_ratings = None
    rating_elem = soup.select_one("span.StarRating-module__srOnly--FAzEA")
    if rating_elem:
        rating_text = rating_elem.get_text(separator=" ", strip=True)
        parts = rating_text.split()
        if len(parts) >= 3:
            rating_value = parts[1]
            number_of_ratings = parts[-2]
    
    # Extract price
    product_price = None
    meta_price = soup.find("meta", {"property": "product:price:amount"})
    if meta_price and meta_price.has_attr("content"):
        product_price = meta_price["content"]
    
    # Extract categories/keywords
    grade_level = None
    grade_elem = soup.select_one('[data-testid="RebrandedContentText"] .NotLinkedSection span')
    if grade_elem:
        grade_level = grade_elem.get_text(strip=True)
    
    categories = []
    category_elems = soup.select('div[data-testid="LabeledSectionContent"] a.Link-module__link--GFbUH')
    if category_elems:
        categories = [cat.get_text(strip=True) for cat in category_elems]
    
    preview_keywords = ""
    if grade_level or categories:
        preview_keywords = (grade_level or "") + (" " if grade_level and categories else "") + ", ".join(categories)
    
    return (title, short_description, long_description, rating_value, number_of_ratings, 
            product_price, preview_keywords, url)

async def scrape_metadata(config_name: str, db_file: str, concurrent_requests: int = 25):
    """
    Scrape metadata for all URLs found in search.
    Only scrapes URLs that haven't been scraped yet.
    """
    logging.info(f"=" * 60)
    logging.info(f"SCRAPE METADATA WORKFLOW - Configuration: {config_name}")
    logging.info(f"=" * 60)
    
    # Get URLs that need metadata
    async with aiosqlite.connect(db_file) as db:
        async with db.execute("""
            SELECT DISTINCT s.url
            FROM search_results s
            LEFT JOIN product_metadata m ON s.url = m.url
            WHERE m.url IS NULL
        """) as cursor:
            urls_to_scrape = [row[0] for row in await cursor.fetchall()]
    
    if not urls_to_scrape:
        logging.info("All URLs already have metadata!")
        return
    
    logging.info(f"URLs to scrape: {len(urls_to_scrape):,}")
    
    # Initialize components
    rate_limiter = AdaptiveRateLimiter(initial_delay=1.0, max_delay=30.0)
    semaphore = asyncio.Semaphore(concurrent_requests)
    
    connector = aiohttp.TCPConnector(
        limit=concurrent_requests + 10,
        limit_per_host=concurrent_requests,
        keepalive_timeout=30
    )
    timeout = aiohttp.ClientTimeout(total=60, connect=10)
    
    async def scrape_with_sem(url):
        async with semaphore:
            return await scrape_product_metadata(session, url, rate_limiter)
    
    async with aiohttp_client_cache.CachedSession(
        cache_name="aiohttp_cache",
        expire_after=3600,
        connector=connector,
        timeout=timeout
    ) as session:
        
        batch_size = 100
        scraped_count = 0
        
        for i in range(0, len(urls_to_scrape), batch_size):
            batch = urls_to_scrape[i:i+batch_size]
            logging.info(f"Scraping batch {i//batch_size + 1}/{(len(urls_to_scrape) + batch_size - 1)//batch_size}")
            
            tasks = [scrape_with_sem(url) for url in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Store results
            async with aiosqlite.connect(db_file) as db:
                for result in results:
                    if result and not isinstance(result, Exception):
                        await db.execute("""
                            INSERT OR IGNORE INTO product_metadata
                            (url, title, short_description, long_description, rating_value, 
                             number_of_ratings, product_price, preview_keywords)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, result)
                        scraped_count += 1
                await db.commit()
            
            logging.info(f"Progress: {min(i + batch_size, len(urls_to_scrape))}/{len(urls_to_scrape)}")
            
            # Sleep between batches
            if i + batch_size < len(urls_to_scrape):
                await asyncio.sleep(random.uniform(1, 3))
    
    logging.info(f"=" * 60)
    logging.info(f"SCRAPING COMPLETE")
    logging.info(f"Metadata scraped for {scraped_count:,} products")
    logging.info(f"=" * 60)

###########################
# 3. DOWNLOAD FREE FILES  #
###########################

async def download_free_file(product_url: str, session_file: str = "tpt_storage.json", 
                            downloads_dir: Path = Path("downloads"), max_retries: int = 3):
    """Download a free file from TPT."""
    free_url = product_url.replace("/Product/", "/FreeDownload/")
    downloads_dir.mkdir(exist_ok=True)
    
    for attempt in range(1, max_retries + 1):
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    storage_state=session_file,
                    accept_downloads=True
                )
                page = await context.new_page()
                
                logging.info(f"Attempt {attempt}: Downloading {free_url}")
                async with page.expect_download() as download_info:
                    await page.goto(free_url)
                
                download = await download_info.value
                suggested_name = download.suggested_filename
                
                # Use URL hash for unique filename
                url_hash = abs(hash(product_url)) % (10 ** 8)
                save_path = downloads_dir / f"{url_hash}_{suggested_name}"
                await download.save_as(save_path)
                
                # Verify file
                if not save_path.exists() or save_path.stat().st_size == 0:
                    raise Exception("Downloaded file is empty or missing")
                
                file_size = save_path.stat().st_size
                logging.info(f"✅ Downloaded: {save_path.name} ({file_size:,} bytes)")
                
                await browser.close()
                return str(save_path), file_size
                
        except Exception as e:
            logging.warning(f"Attempt {attempt} failed: {e}")
            if attempt == max_retries:
                logging.error(f"Failed after {max_retries} attempts")
            await asyncio.sleep(2)
    
    return None, None

async def download_free_products(config_name: str, db_file: str, 
                                filters: Optional[Dict[str, Any]] = None,
                                session_file: str = "tpt_storage.json",
                                concurrent_downloads: int = 5,
                                use_queue: bool = False):
    """
    Download all free products found in search that match filters.
    
    Args:
        config_name: Configuration name
        db_file: Database file
        filters: Optional filters (e.g., {"resource_type": "teacher-tools", "subject": "classroom-management"})
        session_file: Playwright session file for authentication
        concurrent_downloads: Number of concurrent downloads
        use_queue: If True, only download products in the download_queue table
    """
    logging.info(f"=" * 60)
    logging.info(f"DOWNLOAD FREE FILES WORKFLOW - Configuration: {config_name}")
    if use_queue:
        logging.info(f"MODE: Using download_queue table")
    logging.info(f"=" * 60)
    
    # Build query based on mode
    if use_queue:
        # Download from manually populated queue
        query = """
            SELECT DISTINCT q.product_url, m.product_price
            FROM download_queue q
            JOIN product_metadata m ON q.product_url = m.url
            LEFT JOIN downloads d ON q.product_url = d.product_url
            WHERE d.product_url IS NULL
            ORDER BY q.priority DESC, q.added_at ASC
        """
        params = []
    else:
        # Original behavior - download all free products matching filters
        query = """
            SELECT DISTINCT s.url, m.product_price
            FROM search_results s
            JOIN product_metadata m ON s.url = m.url
            LEFT JOIN downloads d ON s.url = d.product_url
            WHERE s.price_option = 'free'
            AND d.product_url IS NULL
        """
        
        params = []
        if filters:
            conditions = []
            for key, value in filters.items():
                if isinstance(value, list):
                    placeholders = ','.join(['?' for _ in value])
                    conditions.append(f"s.{key} IN ({placeholders})")
                    params.extend(value)
                else:
                    conditions.append(f"s.{key} = ?")
                    params.append(value)
            
            if conditions:
                query += " AND " + " AND ".join(conditions)
    
    # Get URLs to download
    async with aiosqlite.connect(db_file) as db:
        async with db.execute(query, params) as cursor:
            urls_to_download = [(row[0], row[1]) for row in await cursor.fetchall()]
    
    if not urls_to_download:
        logging.info("No free products to download!")
        return
    
    logging.info(f"Products to download: {len(urls_to_download):,}")
    if filters:
        logging.info(f"Filters applied: {filters}")
    
    # Download
    semaphore = asyncio.Semaphore(concurrent_downloads)
    downloads_dir = Path(f"downloads_{config_name}")
    
    async def download_with_sem(url):
        async with semaphore:
            return await download_free_file(url, session_file, downloads_dir)
    
    success_count = 0
    fail_count = 0
    
    for i, (url, price) in enumerate(urls_to_download, 1):
        logging.info(f"Processing {i}/{len(urls_to_download)}: {url}")
        
        file_path, file_size = await download_with_sem(url)
        
        if file_path:
            # Store in database
            async with aiosqlite.connect(db_file) as db:
                await db.execute("""
                    INSERT OR IGNORE INTO downloads (product_url, file_path, file_size)
                    VALUES (?, ?, ?)
                """, (url, file_path, file_size))
                await db.commit()
            success_count += 1
        else:
            fail_count += 1
        
        # Small delay between downloads
        if i < len(urls_to_download):
            await asyncio.sleep(random.uniform(2, 5))
    
    logging.info(f"=" * 60)
    logging.info(f"DOWNLOAD COMPLETE")
    logging.info(f"Successful: {success_count:,}")
    logging.info(f"Failed: {fail_count:,}")
    logging.info(f"Files saved to: {downloads_dir}")
    logging.info(f"=" * 60)

###########################
# CLI Interface           #
###########################

async def cmd_config_create(args):
    """Create a new configuration."""
    manager = ConfigManager()
    
    # Load template or start fresh
    if args.template:
        template_path = Path(args.template)
        if not template_path.exists():
            logging.error(f"Template file not found: {args.template}")
            return
        with open(template_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
    else:
        # Minimal default config
        config_data = {
            "resource_type": [""],
            "grade_level": [""],
            "subject": [""],
            "format": [""],
            "price_options": ["free"],
            "supports": [""],
            "sorting_methods": ["Relevance"],
            "total_pages": 10,
            "concurrent_requests": 25
        }
    
    success = manager.create_config(args.name, config_data, args.description or "")
    if success:
        print(f"✅ Created configuration: {args.name}")
        print(f"   Database: {manager.get_db_file(args.name)}")
        print(f"   Edit: configs/{args.name}.json")
    else:
        print(f"❌ Failed to create configuration")

async def cmd_config_list(args):
    """List all configurations."""
    manager = ConfigManager()
    configs = manager.list_configs()
    
    if not configs:
        print("No configurations found.")
        return
    
    print(f"\n{'Name':<20} {'Created':<20} {'Description':<40}")
    print("-" * 80)
    for cfg in configs:
        created = cfg['created'][:10] if len(cfg['created']) > 10 else cfg['created']
        desc = cfg['description'][:40] if cfg['description'] else ""
        print(f"{cfg['name']:<20} {created:<20} {desc:<40}")
    print()

async def cmd_search(args):
    """Run search workflow."""
    manager = ConfigManager()
    config = manager.get_config(args.config)
    if not config:
        logging.error(f"Configuration '{args.config}' not found")
        return
    
    db_file = manager.get_db_file(args.config)
    await setup_db(db_file)
    await search_urls(args.config, config, db_file)

async def cmd_scrape(args):
    """Run scrape metadata workflow."""
    manager = ConfigManager()
    config = manager.get_config(args.config)
    if not config:
        logging.error(f"Configuration '{args.config}' not found")
        return
    
    db_file = manager.get_db_file(args.config)
    await setup_db(db_file)
    await scrape_metadata(args.config, db_file, config.get("concurrent_requests", 25))

async def cmd_download(args):
    """Run download free files workflow."""
    manager = ConfigManager()
    config = manager.get_config(args.config)
    if not config:
        logging.error(f"Configuration '{args.config}' not found")
        return
    
    # Parse filters
    filters = {}
    if args.filter:
        for filter_str in args.filter:
            if '=' in filter_str:
                key, value = filter_str.split('=', 1)
                filters[key] = value
    
    db_file = manager.get_db_file(args.config)
    await setup_db(db_file)
    await download_free_products(args.config, db_file, filters, args.session_file, 
                                args.concurrent, args.use_queue)

async def cmd_stats(args):
    """Show statistics for a configuration."""
    manager = ConfigManager()
    db_file = manager.get_db_file(args.config)
    
    if not Path(db_file).exists():
        logging.error(f"Database not found for configuration '{args.config}'")
        return
    
    async with aiosqlite.connect(db_file) as db:
        # Search results
        async with db.execute("SELECT COUNT(DISTINCT url) FROM search_results") as cursor:
            urls_found = (await cursor.fetchone())[0]
        
        # Metadata scraped
        async with db.execute("SELECT COUNT(*) FROM product_metadata") as cursor:
            metadata_count = (await cursor.fetchone())[0]
        
        # Downloads
        async with db.execute("SELECT COUNT(*), SUM(file_size) FROM downloads") as cursor:
            row = await cursor.fetchone()
            downloads_count = row[0]
            total_size = row[1] or 0
        
        # Free products available
        async with db.execute("""
            SELECT COUNT(DISTINCT s.url)
            FROM search_results s
            JOIN product_metadata m ON s.url = m.url
            WHERE s.price_option = 'free'
        """) as cursor:
            free_products = (await cursor.fetchone())[0]
    
    print(f"\n{'='*60}")
    print(f"STATISTICS - Configuration: {args.config}")
    print(f"{'='*60}")
    print(f"URLs found in search:     {urls_found:,}")
    print(f"Metadata scraped:         {metadata_count:,}")
    print(f"Free products available:  {free_products:,}")
    print(f"Files downloaded:         {downloads_count:,}")
    print(f"Total download size:      {total_size:,} bytes ({total_size / 1024 / 1024:.1f} MB)")
    print(f"{'='*60}\n")

async def main():
    parser = argparse.ArgumentParser(
        description="TPT Scraper - Search, scrape metadata, and download free products",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a new configuration
  python tpt_scraper_refactored.py config create classroom-mgmt --template config.json
  
  # Search for products
  python tpt_scraper_refactored.py search classroom-mgmt
  
  # Scrape metadata
  python tpt_scraper_refactored.py scrape classroom-mgmt
  
  # Download free products
  python tpt_scraper_refactored.py download classroom-mgmt
  
  # Download with filters
  python tpt_scraper_refactored.py download classroom-mgmt --filter resource_type=teacher-tools
  
  # View statistics
  python tpt_scraper_refactored.py stats classroom-mgmt
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Config management
    config_parser = subparsers.add_parser("config", help="Manage configurations")
    config_sub = config_parser.add_subparsers(dest="config_cmd", required=True)
    
    create_parser = config_sub.add_parser("create", help="Create new configuration")
    create_parser.add_argument("name", help="Configuration name")
    create_parser.add_argument("--template", help="Path to template config JSON file")
    create_parser.add_argument("--description", help="Description of this configuration")
    
    config_sub.add_parser("list", help="List all configurations")
    
    # Search
    search_parser = subparsers.add_parser("search", help="Search for product URLs")
    search_parser.add_argument("config", help="Configuration name")
    
    # Scrape metadata
    scrape_parser = subparsers.add_parser("scrape", help="Scrape product metadata")
    scrape_parser.add_argument("config", help="Configuration name")
    
    # Download
    download_parser = subparsers.add_parser("download", help="Download free products")
    download_parser.add_argument("config", help="Configuration name")
    download_parser.add_argument("--filter", action="append", help="Filter (e.g., resource_type=teacher-tools)")
    download_parser.add_argument("--session-file", default="tpt_storage.json", help="Playwright session file")
    download_parser.add_argument("--concurrent", type=int, default=5, help="Concurrent downloads")
    download_parser.add_argument("--use-queue", action="store_true", help="Download only products in download_queue table")
    
    # Stats
    stats_parser = subparsers.add_parser("stats", help="Show statistics")
    stats_parser.add_argument("config", help="Configuration name")
    
    args = parser.parse_args()
    
    # Route to appropriate command
    if args.command == "config":
        if args.config_cmd == "create":
            await cmd_config_create(args)
        elif args.config_cmd == "list":
            await cmd_config_list(args)
    elif args.command == "search":
        await cmd_search(args)
    elif args.command == "scrape":
        await cmd_scrape(args)
    elif args.command == "download":
        await cmd_download(args)
    elif args.command == "stats":
        await cmd_stats(args)

if __name__ == "__main__":
    asyncio.run(main())
