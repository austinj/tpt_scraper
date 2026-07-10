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
from tqdm import tqdm
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
                product_id TEXT,
                title TEXT,
                short_description TEXT,
                long_description TEXT,
                rating_value TEXT,
                number_of_ratings TEXT,
                product_price TEXT,
                preview_keywords TEXT,
                author_name TEXT,
                author_store_url TEXT,
                author_follower_count INTEGER,
                oldest_review_date TEXT,
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Migration: Add new columns if they don't exist (for existing databases)
        async with db.execute("PRAGMA table_info(product_metadata)") as cursor:
            existing_columns = {row[1] for row in await cursor.fetchall()}
        
        migrations = [
            ("author_name", "TEXT"),
            ("author_store_url", "TEXT"),
            ("author_follower_count", "INTEGER"),
            ("product_id", "TEXT"),
            ("oldest_review_date", "TEXT"),
        ]
        
        for col_name, col_type in migrations:
            if col_name not in existing_columns:
                await db.execute(f"ALTER TABLE product_metadata ADD COLUMN {col_name} {col_type}")
                logging.info(f"Migration: Added column '{col_name}' to product_metadata")
        
        # Create index on product_id for faster lookups
        await db.execute("CREATE INDEX IF NOT EXISTS idx_metadata_product_id ON product_metadata(product_id);")
        
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

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


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
                async with session.get(url, headers=BROWSER_HEADERS) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        if rate_limiter:
                            rate_limiter.record_response(True, response_time)
                        return await response.text()
                    elif response.status == 429:
                        if rate_limiter:
                            rate_limiter.record_response(False, response_time)
                        wait = int(response.headers.get("Retry-After", backoff * 4))
                        logging.warning(f"Rate limited (429) — waiting {wait}s")
                        await asyncio.sleep(wait)
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
    concurrent_requests = config.get("concurrent_requests", 8)
    batch_size = config.get("batch_size", 100)
    sleep_min, sleep_max = config.get("sleep_between_batches", [1.0, 3.0])
    max_rate_limit_delay = config.get("max_rate_limit_delay", 30.0)
    
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
    rate_limiter = AdaptiveRateLimiter(initial_delay=1.0, max_delay=max_rate_limit_delay)
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
        batch_size = config.get("batch_size", 100)  # use config value
        for i in range(0, len(remaining), batch_size):
            batch = remaining[i:i+batch_size]
            logging.info(f"Processing batch {i//batch_size + 1}/{(len(remaining) + batch_size - 1)//batch_size}")
            
            tasks = [search_page(combo) for combo in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            total_urls = sum(len(r) for r in results if not isinstance(r, Exception))
            logging.info(f"Found {total_urls} URLs in this batch")
            
            # Sleep between batches
            if i + batch_size < len(remaining):
                await asyncio.sleep(random.uniform(sleep_min, sleep_max))
    
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
    
    # Extract rating from meta tags (more reliable than parsing text)
    rating_value = None
    number_of_ratings = None
    
    meta_rating = soup.find("meta", {"property": "og:rating"})
    if meta_rating and meta_rating.has_attr("content"):
        rating_value = meta_rating["content"]
    
    meta_rating_count = soup.find("meta", {"property": "og:rating_count"})
    if meta_rating_count and meta_rating_count.has_attr("content"):
        number_of_ratings = meta_rating_count["content"]
    
    # Extract price
    product_price = None
    meta_price = soup.find("meta", {"property": "product:price:amount"})
    if meta_price and meta_price.has_attr("content"):
        product_price = meta_price["content"]
    
    # Extract categories/keywords
    grade_level = None
    grade_elem = soup.find(attrs={"data-testid": "GradesLabel"})
    if grade_elem:
        grade_text = grade_elem.get_text(strip=True)
        grade_level = re.sub(r'^Mostly used with\s*', '', grade_text).strip() or None
    
    categories = []
    category_elems = soup.select('[data-testid^="Link_"]')
    if category_elems:
        categories = [cat.get_text(strip=True) for cat in category_elems]
    
    preview_keywords = ""
    if grade_level or categories:
        preview_keywords = (grade_level or "") + (" " if grade_level and categories else "") + ", ".join(categories)
    
    # Extract author/seller info
    author_name = None
    author_store_url = None
    author_follower_count = None
    
    # Find author store link and name via the avatar aria-label
    author_link = soup.find(attrs={"data-testid": "authorAvatarLink"})
    if author_link:
        author_store_url = author_link.get('href', '')
        if not author_store_url.startswith('http'):
            author_store_url = 'https://www.teacherspayteachers.com' + author_store_url
        avatar = author_link.find(attrs={"data-testid": "authorAvatar"})
        if avatar:
            author_name = avatar.get('aria-label', '').strip() or None
    
    # Extract follower count from the AboutAuthorRow follow container
    # The container combines "Follow", count, and "Followers" across child elements,
    # so we get the combined text and parse the number from it.
    # When a seller has 0 followers, the container only shows "Follow" with no count.
    follow_container = soup.find('div', class_=re.compile(r'AboutAuthorRow-module__followContainer'))
    if follow_container:
        container_text = follow_container.get_text(strip=True)
        match = re.search(r'([\d,\.]+)\s*([kKmM])?\s*Followers?', container_text, re.IGNORECASE)
        if match:
            num_str = match.group(1).replace(',', '')
            suffix = match.group(2)
            try:
                num = float(num_str)
                if suffix:
                    suffix = suffix.lower()
                    if suffix == 'k':
                        num *= 1000
                    elif suffix == 'm':
                        num *= 1000000
                author_follower_count = int(num)
            except ValueError:
                author_follower_count = 0
        else:
            # Container exists but no count found — seller has 0 followers
            author_follower_count = 0
    
    # Extract product ID from URL
    product_id = None
    id_match = re.search(r'/Product/[^/]+-(\d+)(?:\?|$)', url)
    if id_match:
        product_id = id_match.group(1)
    
    # Return tuple in order matching INSERT: url, product_id, title, short_description, long_description, 
    # rating_value, number_of_ratings, product_price, preview_keywords, author_name, author_store_url, author_follower_count
    return (url, product_id, title, short_description, long_description, rating_value, number_of_ratings, 
            product_price, preview_keywords, author_name, author_store_url, author_follower_count)

async def scrape_metadata(config_name: str, db_file: str, concurrent_requests: int = 8,
                          rescrape: bool = False, no_cache: bool = False,
                          batch_size: int = 100, sleep_min: float = 1.0, sleep_max: float = 3.0,
                          max_rate_limit_delay: float = 30.0):
    """
    Scrape metadata for all URLs found in search.
    
    Args:
        config_name: Configuration name
        db_file: Database file path
        concurrent_requests: Number of concurrent requests
        rescrape: If True, re-scrape ALL products (not just new ones)
        no_cache: If True, bypass HTTP cache for fresh fetches
    """
    logging.info(f"=" * 60)
    logging.info(f"SCRAPE METADATA WORKFLOW - Configuration: {config_name}")
    if rescrape:
        logging.info(f"MODE: Rescraping ALL products")
    if no_cache:
        logging.info(f"MODE: Cache disabled (fresh fetches)")
    logging.info(f"=" * 60)
    
    # Get URLs to scrape
    async with aiosqlite.connect(db_file) as db:
        if rescrape:
            # Rescrape all URLs from search results
            async with db.execute("SELECT DISTINCT url FROM search_results") as cursor:
                urls_to_scrape = [row[0] for row in await cursor.fetchall()]
        else:
            # Only scrape URLs that don't have metadata yet
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
    
    # Use cached session or regular session based on no_cache flag
    if no_cache:
        session_context = aiohttp.ClientSession(connector=connector, timeout=timeout)
    else:
        session_context = aiohttp_client_cache.CachedSession(
            cache_name="aiohttp_cache",
            expire_after=3600,
            connector=connector,
            timeout=timeout
        )
    
    async with session_context as session:

        scraped_count = 0
        failed_count = 0
        start_time = time.time()
        total_items = len(urls_to_scrape)
        
        for i in range(0, len(urls_to_scrape), batch_size):
            batch = urls_to_scrape[i:i+batch_size]
            logging.info(f"Scraping batch {i//batch_size + 1}/{(len(urls_to_scrape) + batch_size - 1)//batch_size}")
            
            tasks = [scrape_with_sem(url) for url in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Store results
            async with aiosqlite.connect(db_file) as db:
                for result in results:
                    if result and not isinstance(result, Exception):
                        if rescrape:
                            # Upsert: update if exists, insert if not
                            await db.execute("""
                                INSERT INTO product_metadata
                                (url, product_id, title, short_description, long_description, rating_value, 
                                 number_of_ratings, product_price, preview_keywords, author_name,
                                 author_store_url, author_follower_count, scraped_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                                ON CONFLICT(url) DO UPDATE SET
                                    product_id = excluded.product_id,
                                    title = excluded.title,
                                    short_description = excluded.short_description,
                                    long_description = excluded.long_description,
                                    rating_value = excluded.rating_value,
                                    number_of_ratings = excluded.number_of_ratings,
                                    product_price = excluded.product_price,
                                    preview_keywords = excluded.preview_keywords,
                                    author_name = excluded.author_name,
                                    author_store_url = excluded.author_store_url,
                                    author_follower_count = excluded.author_follower_count,
                                    scraped_at = CURRENT_TIMESTAMP
                            """, result)
                        else:
                            await db.execute("""
                                INSERT OR IGNORE INTO product_metadata
                                (url, product_id, title, short_description, long_description, rating_value, 
                                 number_of_ratings, product_price, preview_keywords, author_name,
                                 author_store_url, author_follower_count)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, result)
                        scraped_count += 1
                    else:
                        failed_count += 1
                await db.commit()
            
            # Progress with time tracking
            processed = min(i + batch_size, total_items)
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = total_items - processed
            eta_seconds = remaining / rate if rate > 0 else 0
            
            elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
            eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
            
            logging.info(f"Progress: {processed:,}/{total_items:,} ({processed*100//total_items}%) | OK: {scraped_count:,} | Fail: {failed_count:,} | Elapsed: {elapsed_str} | ETA: {eta_str} | Rate: {rate:.1f}/s")
            
            # Sleep between batches
            if i + batch_size < len(urls_to_scrape):
                await asyncio.sleep(random.uniform(sleep_min, sleep_max))
    
    total_elapsed = time.time() - start_time
    logging.info(f"=" * 60)
    logging.info(f"SCRAPING COMPLETE")
    logging.info(f"Metadata scraped for {scraped_count:,} products")
    logging.info(f"Total time: {int(total_elapsed // 60)}m {int(total_elapsed % 60)}s")
    logging.info(f"=" * 60)

###########################
# 2b. DEEP SCRAPE (Playwright)
###########################

async def deep_scrape_oldest_review(url: str, browser, session_file: str = "tpt_storage.json") -> Tuple[Optional[str], bool]:
    """
    Use Playwright to load a product page, click through all reviews,
    and extract the oldest review date.
    
    Args:
        url: Product URL to scrape
        browser: Playwright browser instance (reused across calls)
        session_file: Path to Playwright storage state
    
    Returns the oldest review date as a string (e.g., "October 28, 2008") or None.
    """
    from datetime import datetime
    
    product_id = extract_product_id(url)
    oldest_date = None
    oldest_datetime = None
    
    try:
        context = await browser.new_context(storage_state=session_file)
        page = await context.new_page()
        
        logging.info(f"[{product_id}] Deep scraping: {url}")
        await page.goto(url, wait_until="domcontentloaded", timeout=45000)
        # Give JS a moment to render review content
        await page.wait_for_timeout(2000)
        
        # Click "Show more reviews" until it's gone or we hit a limit
        max_clicks = 50  # Safety limit
        clicks = 0
        
        while clicks < max_clicks:
            try:
                # Look for the "Show more reviews" button
                show_more = page.locator("text=Show more reviews").first
                if await show_more.is_visible(timeout=2000):
                    await show_more.click()
                    await page.wait_for_timeout(500)  # Wait for content to load
                    clicks += 1
                else:
                    break
            except:
                break
        
        if clicks > 0:
            logging.info(f"[{product_id}] Clicked 'Show more reviews' {clicks} times")
        
        # Now extract all review dates from the page
        # Review dates appear as text like "February 3, 2026", "October 28, 2008", etc.
        content = await page.content()
        
        # Close context to free resources
        await context.close()
        
        # Check if the product actually has reviews via JSON-LD reviewCount/ratingCount
        review_count_match = re.search(r'"reviewCount"\s*:\s*(\d+)', content)
        rating_count_match = re.search(r'"ratingCount"\s*:\s*(\d+)', content)
        review_count = int(review_count_match.group(1)) if review_count_match else 0
        rating_count = int(rating_count_match.group(1)) if rating_count_match else 0
        has_reviews = review_count > 0 or rating_count > 0
        
        if not has_reviews:
            logging.info(f"[{product_id}] No reviews (count=0), skipping date extraction")
            return None, False
        
        # 1. Parse ISO dates from JSON-LD "datePublished" fields (most reliable)
        iso_matches = re.findall(r'"datePublished"\s*:\s*"(\d{4}-\d{2}-\d{2})', content)
        for date_str in iso_matches:
            try:
                parsed = datetime.strptime(date_str, "%Y-%m-%d")
                if oldest_datetime is None or parsed < oldest_datetime:
                    oldest_datetime = parsed
                    oldest_date = date_str
            except ValueError:
                continue
        
        # No text-based fallback — regex on full page HTML matches dates from
        # product descriptions (movie release dates, historical events), Q&A,
        # teacher bios, etc. JSON-LD datePublished is scoped to reviews and
        # is the only reliable source.
        
        if oldest_date:
            logging.info(f"[{product_id}] Oldest review date: {oldest_date} (reviews: {review_count})")
        else:
            logging.warning(f"[{product_id}] Has {review_count} reviews but no dates extracted")
        
        return oldest_date, has_reviews
            
    except Exception as e:
        logging.error(f"[{product_id}] Deep scrape failed: {e}")
        return None, False


async def deep_scrape_products(config_name: str, db_file: str, session_file: str = "tpt_storage.json",
                               limit: Optional[int] = None, concurrent: int = 3):
    """
    Deep scrape products to find oldest review dates using Playwright.
    Only processes products that don't already have an oldest_review_date.
    Uses a single browser instance for efficiency.
    """
    logging.info(f"=" * 60)
    logging.info(f"DEEP SCRAPE WORKFLOW - Configuration: {config_name}")
    logging.info(f"=" * 60)
    
    # Bulk-mark products with 0 ratings as 'no_reviews' so we never visit them
    async with aiosqlite.connect(db_file) as db:
        cursor = await db.execute("""
            UPDATE product_metadata
            SET oldest_review_date = 'no_reviews'
            WHERE oldest_review_date IS NULL
              AND (number_of_ratings IS NULL OR CAST(number_of_ratings AS INTEGER) = 0)
        """)
        bulk_skipped = cursor.rowcount
        await db.commit()
    if bulk_skipped:
        logging.info(f"Bulk-marked {bulk_skipped:,} products with 0 ratings as 'no_reviews'")

    # Get URLs that need deep scraping (have ratings but no oldest_review_date yet)
    async with aiosqlite.connect(db_file) as db:
        query = """
            SELECT url FROM product_metadata 
            WHERE oldest_review_date IS NULL
              AND CAST(number_of_ratings AS INTEGER) > 0
            ORDER BY CAST(product_id AS INTEGER) ASC
        """
        if limit:
            query += f" LIMIT {limit}"
        
        async with db.execute(query) as cursor:
            urls_to_scrape = [row[0] for row in await cursor.fetchall()]
    
    if not urls_to_scrape:
        logging.info("All products already have oldest_review_date!")
        return
    
    logging.info(f"Products to deep scrape: {len(urls_to_scrape):,} (only those with ratings)")
    
    # Launch browser once and reuse for all URLs
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        logging.info("Browser launched (will be reused for all URLs)")
        
        # Process with limited concurrency
        semaphore = asyncio.Semaphore(concurrent)
        scraped_count = 0
        no_reviews_count = 0
        missing_dates_count = 0
        error_count = 0
        start_time = time.time()
        total_items = len(urls_to_scrape)
        
        async def scrape_with_sem(url):
            async with semaphore:
                oldest_date, has_reviews = await deep_scrape_oldest_review(url, browser, session_file)
                return url, oldest_date, has_reviews
        
        # Process in batches to update DB periodically
        batch_size = 10
        
        for i in range(0, len(urls_to_scrape), batch_size):
            batch = urls_to_scrape[i:i+batch_size]
            tasks = [scrape_with_sem(url) for url in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Store results
            async with aiosqlite.connect(db_file) as db:
                for result in results:
                    if isinstance(result, Exception):
                        logging.error(f"Batch error: {result}")
                        error_count += 1
                        continue
                    url, oldest_date, has_reviews = result
                    if oldest_date:
                        await db.execute("""
                            UPDATE product_metadata 
                            SET oldest_review_date = ?
                            WHERE url = ?
                        """, (oldest_date, url))
                        scraped_count += 1
                    elif has_reviews:
                        await db.execute("""
                            UPDATE product_metadata 
                            SET oldest_review_date = 'parse_failed'
                            WHERE url = ?
                        """, (url,))
                        missing_dates_count += 1
                    else:
                        await db.execute("""
                            UPDATE product_metadata 
                            SET oldest_review_date = 'no_reviews'
                            WHERE url = ?
                        """, (url,))
                        no_reviews_count += 1
                await db.commit()
            
            # Progress with time tracking
            processed = min(i + batch_size, total_items)
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = total_items - processed
            eta_seconds = remaining / rate if rate > 0 else 0
            
            elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
            eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
            
            logging.info(f"Progress: {processed:,}/{total_items:,} ({processed*100//total_items}%) | Found: {scraped_count:,} | No reviews: {no_reviews_count:,} | Missing dates: {missing_dates_count:,} | Errors: {error_count:,} | Elapsed: {elapsed_str} | ETA: {eta_str} | Rate: {rate:.1f}/s")
            
            # Small delay between batches
            if i + batch_size < len(urls_to_scrape):
                await asyncio.sleep(1)
        
        await browser.close()
        logging.info("Browser closed")
    
    total_elapsed = time.time() - start_time
    logging.info(f"=" * 60)
    logging.info(f"DEEP SCRAPE COMPLETE")
    logging.info(f"Found oldest review dates for {scraped_count:,} products")
    logging.info(f"Total time: {int(total_elapsed // 60)}m {int(total_elapsed % 60)}s")
    logging.info(f"=" * 60)

###########################
# 3. DOWNLOAD FREE FILES  #
###########################

def extract_product_id(url: str) -> str:
    """Extract product ID from TPT URL. e.g., 'Product-Name-1234567' -> '1234567'"""
    import re
    # URL format: .../Product/Product-Name-1234567
    match = re.search(r'/Product/[^/]+-(\d+)(?:\?|$)', url)
    if match:
        return match.group(1)
    # Fallback to URL hash
    return str(abs(hash(url)) % (10 ** 8))

class SessionExpiredError(Exception):
    """Raised when the TPT session cookie has expired. Aborts the entire download run."""
    pass


async def download_free_file(product_url: str, browser, session_file: str,
                             downloads_dir: Path) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Download a single free product using a shared Playwright browser.

    Returns (file_path, file_size, failure_reason).
    failure_reason is None on success.
    Raises SessionExpiredError if the auth cookie needs to be refreshed.
    """
    free_url = product_url.replace("/Product/", "/FreeDownload/")
    product_id = extract_product_id(product_url)
    product_dir = downloads_dir / product_id
    product_dir.mkdir(parents=True, exist_ok=True)

    context = None
    try:
        context = await browser.new_context(storage_state=session_file, accept_downloads=True)
        page = await context.new_page()

        # Auto-dismiss any modal dialogs that would block the download
        page.on("dialog", lambda d: asyncio.ensure_future(d.dismiss()))

        logging.info(f"[{product_id}] Downloading {free_url}")

        # Attempt 1: direct-download navigation (most products trigger immediately)
        try:
            async with page.expect_download(timeout=15000) as dl_info:
                await page.goto(free_url, wait_until="domcontentloaded", timeout=30000)
            download = await dl_info.value

        except Exception:
            # Download did not auto-start — inspect the landing page
            current_url = page.url

            # Detect session expiry (redirect to login / request-authorization)
            if any(tok in current_url.lower() for tok in
                   ("login", "signin", "sign-in", "request-authorization")):
                raise SessionExpiredError(
                    f"Session expired (redirected to {current_url[:80]}). "
                    "Re-run create_session.py to refresh tpt_storage.json."
                )

            # Attempt 2: look for an explicit Download button on the page
            btn = page.locator(
                '[data-testid="download-button-cta"], [data-testid="download-button"]'
            ).first
            try:
                btn_visible = await btn.is_visible(timeout=3000)
            except Exception:
                btn_visible = False

            if btn_visible:
                logging.info(f"[{product_id}] Clicking download button")
                async with page.expect_download(timeout=30000) as dl_info:
                    await btn.click()
                download = await dl_info.value
            else:
                # Last-resort: check page text for a sign-in wall
                try:
                    body_text = (await page.inner_text("body"))[:400].lower()
                except Exception:
                    body_text = ""
                if "sign in" in body_text or "log in" in body_text:
                    raise SessionExpiredError(
                        "Session expired (sign-in wall detected). "
                        "Re-run create_session.py to refresh tpt_storage.json."
                    )
                return None, None, f"no_download_triggered (landed on {current_url[:80]})"

        filename = download.suggested_filename
        save_path = product_dir / filename
        await download.save_as(save_path)

        if not save_path.exists() or save_path.stat().st_size == 0:
            return None, None, "empty_file"

        file_size = save_path.stat().st_size
        logging.info(f"[{product_id}] Downloaded: {save_path.name} ({file_size:,} bytes)")
        return str(save_path), file_size, None

    except SessionExpiredError:
        raise  # propagate to abort the run
    except Exception as e:
        logging.warning(f"[{product_id}] Download failed: {e}")
        return None, None, str(e)[:200]
    finally:
        if context:
            try:
                await context.close()
            except Exception:
                pass

def extract_zip_in_place(zip_path: Path) -> bool:
    """Extract a zip file in its containing folder and optionally remove the zip."""
    import zipfile
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(zip_path.parent)
        logging.info(f"📦 Extracted: {zip_path.name}")
        return True
    except Exception as e:
        logging.warning(f"Failed to extract {zip_path.name}: {e}")
        return False

async def download_free_products(config_name: str, db_file: str,
                                filters: Optional[Dict[str, Any]] = None,
                                session_file: str = "tpt_storage.json",
                                concurrent_downloads: int = 5,
                                use_queue: bool = False,
                                auto_extract: bool = True):
    """
    Download all free products found in search that match filters.

    Args:
        config_name: Configuration name
        db_file: Database file
        filters: Optional filters (e.g., {"resource_type": "teacher-tools"})
        session_file: Playwright session file for authentication
        concurrent_downloads: Number of concurrent downloads
        use_queue: If True, only download products in the download_queue table
        top_n: If set, only download the top N products by number of ratings
    """
    logging.info(f"=" * 60)
    logging.info(f"DOWNLOAD FREE FILES WORKFLOW - Configuration: {config_name}")
    if use_queue:
        logging.info(f"MODE: Using download_queue table")
    if top_n:
        logging.info(f"MODE: Top {top_n:,} products by rating count")
    logging.info(f"=" * 60)

    downloads_dir = Path(f"downloads_{config_name}")

    # Build query based on mode
    if use_queue:
        # Download from manually populated queue
        query = """
            SELECT DISTINCT q.product_url, m.product_price
            FROM download_queue q
            JOIN product_metadata m ON q.product_url = m.url
            ORDER BY q.priority DESC, q.added_at ASC
        """
        params = []
    else:
        # Download free products matching filters, optionally limited to top N by ratings
        query = """
            SELECT DISTINCT s.url, m.product_price
            FROM search_results s
            JOIN product_metadata m ON s.url = m.url
            WHERE s.price_option = 'free'
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

        query += " ORDER BY CAST(m.number_of_ratings AS INTEGER) DESC"
        if top_n:
            query += f" LIMIT {top_n}"
    
    # Get URLs to download
    async with aiosqlite.connect(db_file) as db:
        async with db.execute(query, params) as cursor:
            all_urls = [(row[0], row[1]) for row in await cursor.fetchall()]
    
    # Filter out already downloaded (check if folder exists)
    urls_to_download = []
    already_downloaded = 0
    for url, price in all_urls:
        product_id = extract_product_id(url)
        product_folder = downloads_dir / product_id
        if product_folder.exists() and any(product_folder.iterdir()):
            already_downloaded += 1
        else:
            urls_to_download.append((url, price))
    
    if already_downloaded > 0:
        logging.info(f"Skipping {already_downloaded:,} already downloaded (folder exists)")
    
    if not urls_to_download:
        logging.info("No free products to download!")
        return
    
    logging.info(f"Products to download: {len(urls_to_download):,}")
    if filters:
        logging.info(f"Filters applied: {filters}")
    
    # DB migration: add failure_reason column if not present
    async with aiosqlite.connect(db_file) as db:
        async with db.execute("PRAGMA table_info(downloads)") as cursor:
            existing_cols = {row[1] for row in await cursor.fetchall()}
        if "failure_reason" not in existing_cols:
            await db.execute("ALTER TABLE downloads ADD COLUMN failure_reason TEXT")
            await db.commit()
            logging.info("Migration: added failure_reason column to downloads")

    # Download using a single shared browser for all products
    success_count = 0
    fail_count = 0
    extracted_count = 0
    session_expired = False
    semaphore = asyncio.Semaphore(concurrent_downloads)
    start_time = time.time()
    total = len(urls_to_download)

    async def download_one(url: str):
        """Download one product with semaphore and session-expiry propagation."""
        nonlocal session_expired
        if session_expired:
            return url, None, None, "aborted_session_expired"
        async with semaphore:
            try:
                file_path, file_size, reason = await download_free_file(
                    url, browser, session_file, downloads_dir
                )
                return url, file_path, file_size, reason
            except SessionExpiredError as e:
                session_expired = True
                logging.error(str(e))
                return url, None, None, "session_expired"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        logging.info("Browser launched (shared across all downloads)")

        batch_size = 10
        for i in range(0, total, batch_size):
            if session_expired:
                remaining_count = total - i
                logging.error(
                    f"Session expired — stopping ({remaining_count} products not downloaded). "
                    "Run create_session.py then re-run download to resume."
                )
                break

            batch_urls = [url for url, _ in urls_to_download[i:i + batch_size]]
            tasks = [download_one(url) for url in batch_urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            async with aiosqlite.connect(db_file) as db:
                for result in results:
                    if isinstance(result, Exception):
                        fail_count += 1
                        continue
                    url, file_path, file_size, failure_reason = result
                    if file_path:
                        await db.execute("""
                            INSERT OR REPLACE INTO downloads
                                (product_url, file_path, file_size, failure_reason)
                            VALUES (?, ?, ?, NULL)
                        """, (url, file_path, file_size))
                        success_count += 1
                        if file_path.endswith(".zip"):
                            if extract_zip_in_place(Path(file_path)):
                                extracted_count += 1
                    else:
                        await db.execute("""
                            INSERT OR REPLACE INTO downloads
                                (product_url, file_path, file_size, failure_reason)
                            VALUES (?, NULL, NULL, ?)
                        """, (url, failure_reason))
                        fail_count += 1
                await db.commit()

            # Progress
            processed = min(i + batch_size, total)
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / rate if rate > 0 else 0
            logging.info(
                f"Progress: {processed:,}/{total:,} ({processed * 100 // total}%) "
                f"| OK: {success_count:,} | Fail: {fail_count:,} "
                f"| Elapsed: {int(elapsed // 60)}m {int(elapsed % 60)}s "
                f"| ETA: {int(eta // 60)}m {int(eta % 60)}s"
            )

            if i + batch_size < total and not session_expired:
                await asyncio.sleep(random.uniform(1, 3))

        await browser.close()
        logging.info("Browser closed")

    logging.info("=" * 60)
    logging.info("DOWNLOAD COMPLETE")
    logging.info(f"Successful: {success_count:,}")
    logging.info(f"Extracted zips: {extracted_count:,}")
    logging.info(f"Failed: {fail_count:,}")
    logging.info(f"Files saved to: {downloads_dir}")
    logging.info("=" * 60)

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
    
    # Auto-start scrape if --auto-scrape flag is set
    if getattr(args, 'auto_scrape', False):
        logging.info("=" * 60)
        logging.info("AUTO-SCRAPE: Starting metadata scrape...")
        logging.info("=" * 60)
        await scrape_metadata(args.config, db_file, config.get("concurrent_requests", 25), False, False)

async def cmd_scrape(args):
    """Run scrape metadata workflow."""
    manager = ConfigManager()
    config = manager.get_config(args.config)
    if not config:
        logging.error(f"Configuration '{args.config}' not found")
        return
    
    db_file = manager.get_db_file(args.config)
    await setup_db(db_file)
    await scrape_metadata(
        args.config, db_file,
        concurrent_requests=config.get("concurrent_requests", 8),
        rescrape=args.rescrape,
        no_cache=args.no_cache,
        batch_size=config.get("batch_size", 100),
        sleep_min=config.get("sleep_between_batches", [1.0, 3.0])[0],
        sleep_max=config.get("sleep_between_batches", [1.0, 3.0])[1],
        max_rate_limit_delay=config.get("max_rate_limit_delay", 30.0),
    )

async def cmd_deepscrape(args):
    """Run deep scrape workflow to find oldest review dates."""
    manager = ConfigManager()
    config = manager.get_config(args.config)
    if not config:
        logging.error(f"Configuration '{args.config}' not found")
        return
    
    db_file = manager.get_db_file(args.config)
    await setup_db(db_file)
    await deep_scrape_products(args.config, db_file, args.session_file, args.limit, args.concurrent)

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
                                args.concurrent, args.use_queue, top_n=args.top)

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
        async with db.execute("SELECT COUNT(*), SUM(file_size) FROM downloads WHERE file_path IS NOT NULL") as cursor:
            row = await cursor.fetchone()
            downloads_count = row[0]
            total_size = row[1] or 0

        async with db.execute("SELECT COUNT(*) FROM downloads WHERE failure_reason IS NOT NULL") as cursor:
            failed_downloads = (await cursor.fetchone())[0]
        
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
    print(f"Download failures:        {failed_downloads:,}")
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
  
  # Deep scrape to find oldest review dates (uses Playwright)
  python tpt_scraper_refactored.py deepscrape classroom-mgmt --limit 100
  
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
    search_parser.add_argument("--auto-scrape", action="store_true", dest="auto_scrape", help="Automatically start scraping metadata after search completes")
    
    # Scrape metadata
    scrape_parser = subparsers.add_parser("scrape", help="Scrape product metadata")
    scrape_parser.add_argument("config", help="Configuration name")
    scrape_parser.add_argument("--rescrape", action="store_true", help="Re-scrape ALL products, updating existing records")
    scrape_parser.add_argument("--no-cache", action="store_true", dest="no_cache", help="Bypass HTTP cache for fresh fetches")
    
    # Deep scrape (Playwright-based)
    deepscrape_parser = subparsers.add_parser("deepscrape", help="Deep scrape to find oldest review dates (uses Playwright)")
    deepscrape_parser.add_argument("config", help="Configuration name")
    deepscrape_parser.add_argument("--session-file", default="tpt_storage.json", help="Playwright session file")
    deepscrape_parser.add_argument("--limit", type=int, help="Limit number of products to deep scrape")
    deepscrape_parser.add_argument("--concurrent", type=int, default=3, help="Concurrent browser instances (default: 3)")
    
    # Download
    download_parser = subparsers.add_parser("download", help="Download free products")
    download_parser.add_argument("config", help="Configuration name")
    download_parser.add_argument("--filter", action="append", help="Filter (e.g., resource_type=teacher-tools)")
    download_parser.add_argument("--session-file", default="tpt_storage.json", help="Playwright session file")
    download_parser.add_argument("--concurrent", type=int, default=3, help="Concurrent downloads")
    download_parser.add_argument("--top", type=int, default=None, metavar="N",
                                 help="Only download the top N products by number of ratings")
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
    elif args.command == "deepscrape":
        await cmd_deepscrape(args)
    elif args.command == "download":
        await cmd_download(args)
    elif args.command == "stats":
        await cmd_stats(args)

if __name__ == "__main__":
    asyncio.run(main())
