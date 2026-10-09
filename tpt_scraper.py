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
        
        # URLs discovered during search (one row per unique URL; first combination that found it)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS search_results (
                url TEXT PRIMARY KEY,
                product_id TEXT,
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
        async with db.execute("PRAGMA table_info(search_results)") as cursor:
            search_columns = {row[1] for row in await cursor.fetchall()}
        if "product_id" not in search_columns:
            await db.execute("ALTER TABLE search_results ADD COLUMN product_id TEXT")
            logging.info("Migration: Added column 'product_id' to search_results")
        # Backfill product_id for rows written before the column existed
        async with db.execute("SELECT url FROM search_results WHERE product_id IS NULL") as cursor:
            missing = [row[0] for row in await cursor.fetchall()]
        if missing:
            await db.executemany(
                "UPDATE search_results SET product_id = ? WHERE url = ?",
                [(extract_product_id(u), u) for u in missing],
            )
            logging.info(f"Migration: backfilled product_id for {len(missing):,} search_results rows")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_search_product_id ON search_results(product_id);")

        # Every (product, search combination, page, rank) appearance. A product that
        # shows up under several grades/subjects/prices/sort orders gets one row per
        # appearance here, which is what lets you measure overlap between searches.
        await db.execute("""
            CREATE TABLE IF NOT EXISTS search_hits (
                product_id TEXT NOT NULL,
                url TEXT NOT NULL,
                resource_type TEXT,
                grade_level TEXT,
                subject TEXT,
                format TEXT,
                price_option TEXT,
                supports TEXT,
                sort_order TEXT,
                page INTEGER,
                rank_on_page INTEGER,
                discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(product_id, resource_type, grade_level, subject, format, price_option, supports, sort_order, page)
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_hits_product_id ON search_hits(product_id);")

        # Per-product overlap summary: how many distinct search combinations surfaced it
        await db.execute("""
            CREATE VIEW IF NOT EXISTS product_search_summary AS
            SELECT
                product_id,
                MIN(url)                                   AS url,
                COUNT(*)                                   AS n_hits,
                COUNT(DISTINCT grade_level)                AS n_grade_levels,
                COUNT(DISTINCT subject)                    AS n_subjects,
                COUNT(DISTINCT price_option)               AS n_price_options,
                COUNT(DISTINCT sort_order)                 AS n_sort_orders,
                MIN(page)                                  AS best_page,
                MIN(page * 1000 + rank_on_page)            AS best_position,
                GROUP_CONCAT(DISTINCT grade_level)         AS grade_levels,
                GROUP_CONCAT(DISTINCT subject)             AS subjects,
                GROUP_CONCAT(DISTINCT price_option)        AS price_options,
                GROUP_CONCAT(DISTINCT sort_order)          AS sort_orders
            FROM search_hits
            GROUP BY product_id
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
            # Fields read from the page's embedded Apollo state (see extract_product_details)
            ("post_date", "TEXT"),
            ("item_type", "TEXT"),
            ("file_kind", "TEXT"),
            ("file_size_bytes", "INTEGER"),
            ("page_count", "TEXT"),
            ("price_full", "TEXT"),
            ("price_discount", "TEXT"),
            ("price_sale", "TEXT"),
            ("price_additional_license", "TEXT"),
            ("is_free", "INTEGER"),
            ("taxonomy_tags", "TEXT"),
            ("common_core_standards", "TEXT"),
            ("answer_key", "TEXT"),
            ("teaching_duration", "TEXT"),
            ("bundle_count", "INTEGER"),
            ("image_count", "INTEGER"),
            ("has_preview_video", "INTEGER"),
            ("has_main_video", "INTEGER"),
            ("store_id", "TEXT"),
            ("store_follower_count", "INTEGER"),
            ("store_total_items", "INTEGER"),
            ("v1_evaluations_count", "INTEGER"),
            ("rating_distribution", "TEXT"),
            ("evaluation_facets", "TEXT"),
            # True first-review date from the evaluations GraphQL API (see fetch_first_review_date)
            ("first_review_date", "TEXT"),
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
    """Adaptive rate limiter that adjusts based on response times and errors.
    Uses a lock so concurrent tasks are dispatched sequentially, not in bursts."""

    def __init__(self, initial_delay=1.0, max_delay=30.0, error_threshold=0.1):
        self.current_delay = initial_delay
        self.max_delay = max_delay
        self.error_threshold = error_threshold
        self.recent_responses = deque(maxlen=100)
        self.error_count = 0
        self.total_count = 0
        self._lock = asyncio.Lock()  # serialises request dispatch

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
        """Acquire the lock, sleep for the current delay, then release.
        This ensures requests are staggered even under high concurrency."""
        async with self._lock:
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

# Product title links on browse pages. CSS-module class names carry a build hash
# ("--YPqiC") that changes on every TPT frontend deploy, so match on the stable
# stem rather than the full class.
PRODUCT_CARD_LINK_SELECTOR = 'a[class*="ProductRowCard-module__cardTitleLink"]'

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


async def fetch(session, url, rate_limiter: Optional[AdaptiveRateLimiter] = None, max_retries=3):
    """Fetch a URL with retry logic."""
    retries = 0
    backoff = 1
    
    while retries < max_retries:
        try:
            if rate_limiter:
                await rate_limiter.wait()
            # Time only the HTTP round-trip: including the limiter's own sleep made the
            # measured response time always exceed the 2s speed-up threshold, so the
            # adaptive limiter could never back off from its initial delay.
            start_time = time.time()
                
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
                        logging.warning(f"Rate limited (429) - waiting {wait}s")
                        await asyncio.sleep(wait)
                        raise Exception(f"Rate limited (429)")
                    elif response.status >= 500:
                        if rate_limiter:
                            rate_limiter.record_response(False, response_time)
                        raise Exception(f"Server error {response.status}")
                    else:
                        if rate_limiter:
                            rate_limiter.record_response(False, response_time)
                        logging.warning(f"Unexpected HTTP {response.status} for {url}")
                        return None

        except Exception as e:
            retries += 1
            logging.debug(f"Fetch attempt {retries}/{max_retries} failed for {url}: {e}")
            if retries < max_retries:
                await asyncio.sleep(backoff)
                backoff *= 2
                
    logging.error(f"Failed to fetch {url} after {max_retries} retries")
    return None

###########################
# Pre-flight check        #
###########################

async def preflight_check(config: Dict[str, Any]) -> Tuple[bool, set]:
    """
    Test one URL per unique base combination (grade×subject×price) before the full search.

    Returns (ok, bad_keys) where:
      - ok=False  → hard failure (403/429/connection error); caller should abort
      - ok=True   → proceed; bad_keys is the set of base-combo tuples to skip
    """
    resource_types = config.get("resource_type", [""])
    grade_levels   = config.get("grade_level", [""])
    subjects       = config.get("subject", [""])
    formats        = config.get("format", [""])
    price_options  = config.get("price_options", [""])
    supports_list  = config.get("supports", [""])
    sort           = config.get("sorting_methods", ["Relevance"])[0]

    probe_urls: Dict[tuple, str] = {}
    for rt in resource_types:
        for gl in grade_levels:
            for sub in subjects:
                for fmt in formats:
                    for price in price_options:
                        for sup in supports_list:
                            key = (rt, gl, sub, fmt, price, sup)
                            probe_urls[key] = build_page_url(
                                rt, gl, sub, fmt, price, sup, sort, 1
                            )

    logging.info(f"Pre-flight: testing {len(probe_urls)} unique URL combinations...")

    HARD_FAIL = {403, 429, 503}
    bad_keys: set = set()
    SELECTOR = PRODUCT_CARD_LINK_SELECTOR

    async with aiohttp.ClientSession() as session:
        for key, url in probe_urls.items():
            try:
                async with async_timeout.timeout(15):
                    async with session.get(
                        url, headers=BROWSER_HEADERS, allow_redirects=True
                    ) as resp:
                        status = resp.status
                        if status in HARD_FAIL:
                            logging.error(
                                f"Pre-flight HARD FAIL HTTP {status}: {url}\n"
                                f"  Looks like this IP is rate-limited or blocked. "
                                f"Wait a few minutes then try again."
                            )
                            return False, set()
                        elif status == 404:
                            logging.warning(f"Pre-flight 404 - skipping: {url}")
                            bad_keys.add(key)
                        elif status == 200:
                            html = await resp.text()
                            count = len(BeautifulSoup(html, "lxml").select(SELECTOR))
                            if count == 0:
                                logging.warning(
                                    f"Pre-flight 200 but 0 products - skipping: {url}"
                                )
                                bad_keys.add(key)
                            else:
                                logging.info(
                                    f"Pre-flight OK ({count} products): {url}"
                                )
                        else:
                            logging.warning(
                                f"Pre-flight HTTP {status} - skipping: {url}"
                            )
                            bad_keys.add(key)
                        await asyncio.sleep(0.5)
            except Exception as e:
                logging.error(
                    f"Pre-flight connection error: {e}\n"
                    f"  URL: {url}\n"
                    f"  Check network and try again."
                )
                return False, set()

    good = len(probe_urls) - len(bad_keys)
    logging.info(
        f"Pre-flight complete: {good}/{len(probe_urls)} valid"
        + (f", {len(bad_keys)} skipped (404/empty)" if bad_keys else "")
    )

    if good == 0:
        logging.error("Pre-flight: no valid URL combinations. Check your config.")
        return False, set()

    return True, bad_keys


# Sort orders TPT's browse UI currently offers. Unknown values are not rejected by
# the server - they silently fall back to Relevance, duplicating those requests.
KNOWN_SORT_ORDERS = {"Relevance", "Rating", "Price-Asc", "Most-Recent"}


async def check_sort_orders(config: Dict[str, Any], good_keys: List[tuple]) -> List[str]:
    """
    Return the config's sort orders minus any the server ignores.

    Fetches page 1 of one valid base combination under each sort order and compares
    the product sequence to the default (Relevance) ordering; an identical sequence
    means the value is being ignored. Falls back to the configured list on error.
    """
    sort_orders = config.get("sorting_methods", ["Relevance"])
    if len(sort_orders) <= 1 or not good_keys:
        return sort_orders

    unknown = [s for s in sort_orders if s not in KNOWN_SORT_ORDERS]
    if unknown:
        logging.warning(f"Sort orders not in the current TPT UI: {unknown} - verifying against the server")

    rt, gl, sub, fmt, price, sup = good_keys[0]

    def product_ids(html: str) -> List[str]:
        soup = BeautifulSoup(html, "lxml")
        return [extract_product_id(a.get("href", "")) or a.get("href", "")
                for a in soup.select(PRODUCT_CARD_LINK_SELECTOR)]

    sequences: Dict[str, List[str]] = {}
    try:
        async with aiohttp.ClientSession() as session:
            for sort in dict.fromkeys(["Relevance"] + list(sort_orders)):
                url = build_page_url(rt, gl, sub, fmt, price, sup, sort, 1)
                async with async_timeout.timeout(15):
                    async with session.get(url, headers=BROWSER_HEADERS, allow_redirects=True) as resp:
                        if resp.status != 200:
                            logging.warning(f"Sort check: HTTP {resp.status} for {url}; skipping sort validation")
                            return sort_orders
                        sequences[sort] = product_ids(await resp.text())
                await asyncio.sleep(0.5)
    except Exception as e:
        logging.warning(f"Sort check failed ({e}); using configured sort orders as-is")
        return sort_orders

    baseline = sequences.get("Relevance") or []
    kept, dropped = [], []
    for sort in sort_orders:
        if sort != "Relevance" and baseline and sequences.get(sort) == baseline:
            dropped.append(sort)
        else:
            kept.append(sort)
    if dropped:
        logging.warning(
            f"Sort check: {dropped} return the same results as Relevance (server ignores them) - "
            f"dropping to avoid duplicate requests. Valid sorts: {sorted(KNOWN_SORT_ORDERS)}"
        )
    else:
        logging.info(f"Sort check: all {len(sort_orders)} sort orders produce distinct orderings")
    return kept


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

    # Pre-flight: verify base URLs before committing to the full run
    ok, bad_keys = await preflight_check(config)
    if not ok:
        logging.error("Search aborted due to pre-flight failure.")
        return

    # Drop sort orders the server ignores (they would just re-fetch the Relevance pages)
    good_keys = [k for k in dict.fromkeys((c[0], c[1], c[2], c[3], c[4], c[5]) for c in combinations)
                 if k not in bad_keys]
    sorting_methods = await check_sort_orders(config, good_keys)

    before = len(combinations)
    combinations = [
        c for c in combinations
        if (c[0], c[1], c[2], c[3], c[4], c[5]) not in bad_keys and c[6] in sorting_methods
    ]
    if before != len(combinations):
        logging.info(
            f"Pre-flight filtered {before - len(combinations):,} combinations "
            f"({len(combinations):,} remaining)"
        )
    
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
            product_elements = soup.select(PRODUCT_CARD_LINK_SELECTOR)
            
            urls_found = []
            for rank, product in enumerate(product_elements, start=1):
                href = product.get("href")
                if href:
                    full_url = f"https://www.teacherspayteachers.com{href}" if not href.startswith("http") else href
                    urls_found.append((
                        full_url, extract_product_id(full_url), resource_type, grade_level, subject, format_type,
                        price_option, supports_val, sort_order, page, rank
                    ))
            
            # Store results
            async with aiosqlite.connect(db_file) as db:
                for (full_url, product_id, *combo_fields, rank) in urls_found:
                    # First-seen record (unique per URL) - downstream stages read this table
                    await db.execute("""
                        INSERT OR IGNORE INTO search_results 
                        (url, product_id, resource_type, grade_level, subject, format, price_option, supports, sort_order, page)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (full_url, product_id, *combo_fields))
                    # Every appearance, for overlap analysis
                    if product_id:
                        await db.execute("""
                            INSERT OR IGNORE INTO search_hits
                            (product_id, url, resource_type, grade_level, subject, format, price_option, supports, sort_order, page, rank_on_page)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (product_id, full_url, *combo_fields, rank))
                
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


def extract_product_id(url: str) -> Optional[str]:
    """Extract the numeric product ID from a TPT product URL, e.g.
    '.../Product/Some-Title-1234567' or '.../Product/--1234567' -> '1234567'."""
    match = re.search(r'/Product/[^/?#]*?-(\d+)/?(?:[?#].*)?$', url)
    return match.group(1) if match else None


# Columns written by scrape_metadata(), in INSERT order. Every dict returned by
# scrape_product_metadata() has exactly these keys.
METADATA_COLUMNS = [
    "url", "product_id", "title", "short_description", "long_description",
    "rating_value", "number_of_ratings", "product_price", "preview_keywords",
    "author_name", "author_store_url", "author_follower_count",
    # Apollo-state fields
    "post_date", "item_type", "file_kind", "file_size_bytes", "page_count",
    "price_full", "price_discount", "price_sale", "price_additional_license", "is_free",
    "taxonomy_tags", "common_core_standards", "answer_key", "teaching_duration",
    "bundle_count", "image_count", "has_preview_video", "has_main_video",
    "store_id", "store_follower_count", "store_total_items",
    "v1_evaluations_count", "rating_distribution", "evaluation_facets",
]
PRODUCT_DETAIL_COLUMNS = METADATA_COLUMNS[METADATA_COLUMNS.index("post_date"):]


def extract_apollo_state(html: str) -> Optional[dict]:
    """Return the Apollo client cache that TPT embeds in every server-rendered
    product page ("apolloState": {...}). It holds the normalized GraphQL objects
    (Product, Store, Evaluation facets, ...) the page was rendered from."""
    marker = '"apolloState":'
    start = html.find(marker)
    if start < 0:
        return None
    try:
        state, _ = json.JSONDecoder().raw_decode(html, start + len(marker))
    except ValueError:
        return None
    return state if isinstance(state, dict) else None


def _facet_counts(facet_list) -> dict:
    return {f.get("value"): f.get("count") for f in facet_list or [] if isinstance(f, dict)}


def extract_product_details(state: Optional[dict], product_id: Optional[str]) -> dict:
    """Pull product/store/review-facet fields out of the Apollo state.
    Always returns a dict with every PRODUCT_DETAIL_COLUMNS key (None if unavailable)."""
    details = {col: None for col in PRODUCT_DETAIL_COLUMNS}
    if not state:
        return details

    product = state.get(f"Product:{product_id}") if product_id else None
    if not isinstance(product, dict):
        # Bundles reference sibling Product:* entries; pick the one the page is about
        product = next((v for k, v in state.items()
                        if k.startswith("Product:") and isinstance(v, dict) and "postDate" in v), None)
    if not isinstance(product, dict):
        return details

    details["post_date"] = product.get("postDate")
    details["item_type"] = product.get("itemType")
    details["file_kind"] = product.get("kind")

    file_preview = product.get("filePreview") or {}
    details["file_size_bytes"] = file_preview.get("filesize")
    details["page_count"] = file_preview.get("pageCount")

    prices = product.get("prices") or {}
    license_prices = prices.get("nonTransferableLicense") or {}
    details["price_full"] = license_prices.get("price") or product.get("price")
    details["price_discount"] = license_prices.get("discountPrice") or product.get("discountprice")
    details["price_sale"] = license_prices.get("salePrice") or product.get("saleprice")
    details["price_additional_license"] = license_prices.get("additionalLicensePrice") or product.get("licenseprice")
    is_free = prices.get("isFree", product.get("isFree"))
    details["is_free"] = None if is_free is None else int(bool(is_free))

    tags = [ref["__ref"].split(":", 1)[1] for ref in product.get("taxonomyTags") or []
            if isinstance(ref, dict) and isinstance(ref.get("__ref"), str) and ":" in ref["__ref"]]
    details["taxonomy_tags"] = ", ".join(tags) if tags else None

    standards = product.get("commonCoreStandards") or []
    details["common_core_standards"] = json.dumps(standards) if standards else None
    details["answer_key"] = product.get("answerKey")
    details["teaching_duration"] = product.get("teachingDuration")
    details["bundle_count"] = len(product.get("includedIn") or [])
    details["image_count"] = len(product.get("images") or [])
    details["has_preview_video"] = int(bool(product.get("previewVideo")))
    details["has_main_video"] = int(bool(product.get("mainVideo")))
    details["v1_evaluations_count"] = (product.get("evaluationRating") or {}).get("v1EvaluationsCount")

    store_ref = (product.get("author") or {}).get("__ref")
    store = state.get(store_ref) if store_ref else None
    if isinstance(store, dict):
        details["store_id"] = store.get("id")
        details["store_follower_count"] = store.get("followerCount")
        details["store_total_items"] = store.get("totalItems")

    # Review facets (star distribution, grades used with, SPED use, yes/no signals)
    # live on the cached filterEvaluationsByResource query under ROOT_QUERY.
    root = state.get("ROOT_QUERY") or {}
    for key, value in root.items():
        if key.startswith("filterEvaluationsByResource(") and isinstance(value, dict):
            facets = value.get("facets") or {}
            star = _facet_counts(facets.get("starRatingGroup"))
            details["rating_distribution"] = json.dumps(star) if star else None
            all_facets = {name: _facet_counts(lst) for name, lst in facets.items() if isinstance(lst, list)}
            details["evaluation_facets"] = json.dumps(all_facets) if all_facets else None
            break

    return details


async def scrape_product_metadata(session, url, rate_limiter: Optional[AdaptiveRateLimiter] = None):
    """Scrape metadata from a single product URL. Returns a dict keyed by METADATA_COLUMNS."""
    html = await fetch(session, url, rate_limiter)
    if not html:
        logging.warning(f"[{extract_product_id(url) or '?'}] FAILED fetch: {url}")
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
        # The label is several adjacent nodes ("K-8th", "Mostly used with", "6th") that
        # get_text(strip=True) fuses into "K-8th Mostly used with6th". Split them into
        # two comma-separated keywords so the tokenizer sees the range and the
        # "mostly used with" grades separately.
        grade_text = grade_elem.get_text(" ", strip=True)
        grade_text = re.sub(r'\s*Mostly used with\s*', ', mostly used with ', grade_text, flags=re.IGNORECASE)
        grade_text = re.sub(r'\s+', ' ', grade_text).strip(' ,')
        grade_level = grade_text or None
    
    categories = []
    category_elems = soup.select('[data-testid^="Link_"]')
    if category_elems:
        categories = [cat.get_text(strip=True) for cat in category_elems]
    
    preview_keywords = ", ".join([k for k in [grade_level] if k] + categories)
    
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
            # Container exists but no count found - seller has 0 followers
            author_follower_count = 0
    
    # Extract product ID from URL
    product_id = extract_product_id(url)

    record = {
        "url": url, "product_id": product_id, "title": title,
        "short_description": short_description, "long_description": long_description,
        "rating_value": rating_value, "number_of_ratings": number_of_ratings,
        "product_price": product_price, "preview_keywords": preview_keywords,
        "author_name": author_name, "author_store_url": author_store_url,
        "author_follower_count": author_follower_count,
    }
    record.update(extract_product_details(extract_apollo_state(html), product_id))
    logging.info(
        f"[{product_id or '?'}] ratings={number_of_ratings or 0} "
        f"posted={(record['post_date'] or 'MISSING')[:10]} "
        f"followers={record['store_follower_count'] if record['store_follower_count'] is not None else '?'} "
        f"price={record['price_full'] or product_price or '?'} | {re.sub(r'\s+', ' ', title or '').strip()[:60]}"
    )
    return record

async def scrape_metadata(config_name: str, db_file: str, concurrent_requests: int = 8,
                          rescrape: bool = False, no_cache: bool = False,
                          batch_size: int = 100, sleep_min: float = 1.0, sleep_max: float = 3.0,
                          max_rate_limit_delay: float = 30.0, backfill: bool = False):
    """
    Scrape metadata for all URLs found in search.
    
    Args:
        config_name: Configuration name
        db_file: Database file path
        concurrent_requests: Number of concurrent requests
        rescrape: If True, re-scrape ALL products (not just new ones)
        no_cache: If True, bypass HTTP cache for fresh fetches
        backfill: If True, re-scrape only existing products that are missing
                  post_date (i.e. scraped before the Apollo-state fields were added).
                  Resumable: each batch is committed as it completes.
    """
    logging.info(f"=" * 60)
    logging.info(f"SCRAPE METADATA WORKFLOW - Configuration: {config_name}")
    if rescrape:
        logging.info(f"MODE: Rescraping ALL products")
    if backfill:
        logging.info(f"MODE: Backfilling products missing post_date")
    if no_cache:
        logging.info(f"MODE: Cache disabled (fresh fetches)")
    logging.info(f"=" * 60)
    
    # Get URLs to scrape
    async with aiosqlite.connect(db_file) as db:
        if rescrape:
            # Rescrape all URLs from search results
            async with db.execute("SELECT DISTINCT url FROM search_results") as cursor:
                urls_to_scrape = [row[0] for row in await cursor.fetchall()]
        elif backfill:
            async with db.execute("SELECT url FROM product_metadata WHERE post_date IS NULL") as cursor:
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

        # Column-driven SQL so new Apollo fields only need adding to METADATA_COLUMNS.
        # Backfill keeps the original preview_keywords: the grade label it came from is
        # client-rendered and not reliably present in a plain fetch, and taxonomy_tags
        # now carries the same information in a cleaner form.
        cols = ", ".join(METADATA_COLUMNS)
        placeholders = ", ".join(f":{c}" for c in METADATA_COLUMNS)
        preserved = {"url"} | ({"preview_keywords"} if backfill and not rescrape else set())
        update_set = ", ".join(f"{c} = excluded.{c}" for c in METADATA_COLUMNS if c not in preserved)
        upsert_sql = f"""
            INSERT INTO product_metadata ({cols}, scraped_at)
            VALUES ({placeholders}, CURRENT_TIMESTAMP)
            ON CONFLICT(url) DO UPDATE SET {update_set}, scraped_at = CURRENT_TIMESTAMP
        """
        insert_sql = f"INSERT OR IGNORE INTO product_metadata ({cols}) VALUES ({placeholders})"
        write_sql = upsert_sql if (rescrape or backfill) else insert_sql
        
        for i in range(0, len(urls_to_scrape), batch_size):
            batch = urls_to_scrape[i:i+batch_size]
            logging.info(f"Scraping batch {i//batch_size + 1}/{(len(urls_to_scrape) + batch_size - 1)//batch_size}")
            
            tasks = [scrape_with_sem(url) for url in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Store results
            async with aiosqlite.connect(db_file) as db:
                for result in results:
                    if result and not isinstance(result, Exception):
                        await db.execute(write_sql, result)
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
# 2b. DEEP SCRAPE (first review date via GraphQL)
###########################

# The product page's reviews widget pages through the evaluations API with
# base64-encoded integer offsets as cursors ("OA==" -> 8). The server caps a
# page at 100 and reports totalCount as at most 10,000, but offsets past 10,000
# still return rows, so jumping to `number_of_ratings - 100` sorted by mostRecent
# lands on the oldest reviews in a single request.
GATEWAY_GRAPHQL_URL = "https://www.teacherspayteachers.com/gateway/graphql"
EVALUATIONS_PAGE_SIZE = 100
EVALUATIONS_MAX_BACKSTEPS = 5    # when number_of_ratings overshoots the real count
EVALUATIONS_MAX_FOLLOW_PAGES = 20  # when new reviews arrived since the metadata scrape
EVALUATIONS_QUERY = """
query FirstReview($resourceId: ID!, $first: Int, $after: String) {
  filterEvaluationsByResource(
    resourceId: $resourceId
    filters: {}
    sortBy: "mostRecent"
    pagination: { first: $first, after: $after }
  ) {
    totalCount
    pageInfo { hasNextPage lastCursor }
    edges { cursor node { createdAt } }
  }
}
"""
GRAPHQL_HEADERS = {
    "User-Agent": BROWSER_HEADERS["User-Agent"],
    "Accept": "*/*",
    "Accept-Language": BROWSER_HEADERS["Accept-Language"],
    "Content-Type": "application/json",
    "Origin": "https://www.teacherspayteachers.com",
    "Referer": "https://www.teacherspayteachers.com/",
}


def encode_evaluations_cursor(offset: int) -> Optional[str]:
    """Cursor for 'rows after offset'. None means start from the beginning."""
    if offset <= 0:
        return None
    import base64
    return base64.b64encode(str(offset).encode()).decode()


async def fetch_evaluations_page(session, product_id: str, after: Optional[str],
                                 rate_limiter: Optional[AdaptiveRateLimiter] = None,
                                 max_retries: int = 3) -> Optional[dict]:
    """POST one page of the evaluations query. Returns the connection dict
    (totalCount / pageInfo / edges) or None after exhausting retries."""
    payload = {
        "query": EVALUATIONS_QUERY,
        "variables": {"resourceId": str(product_id), "first": EVALUATIONS_PAGE_SIZE, "after": after},
    }
    backoff = 1
    for attempt in range(1, max_retries + 1):
        try:
            if rate_limiter:
                await rate_limiter.wait()
            start_time = time.time()  # time the round-trip only (see fetch)
            async with async_timeout.timeout(30):
                async with session.post(GATEWAY_GRAPHQL_URL, json=payload, headers=GRAPHQL_HEADERS) as response:
                    response_time = time.time() - start_time
                    if response.status == 429:
                        if rate_limiter:
                            rate_limiter.record_response(False, response_time)
                        wait = int(response.headers.get("Retry-After", backoff * 4))
                        logging.warning(f"[{product_id}] GraphQL rate limited (429) - waiting {wait}s")
                        await asyncio.sleep(wait)
                        raise Exception("Rate limited (429)")
                    if response.status != 200:
                        if rate_limiter:
                            rate_limiter.record_response(False, response_time)
                        raise Exception(f"HTTP {response.status}")
                    body = await response.json(content_type=None)
            if rate_limiter:
                rate_limiter.record_response(True, response_time)
            if body.get("errors"):
                # Schema-level rejection (bad id, bad argument) - retrying won't help
                logging.warning(f"[{product_id}] GraphQL errors: {json.dumps(body['errors'])[:300]}")
                return None
            connection = (body.get("data") or {}).get("filterEvaluationsByResource")
            return connection if isinstance(connection, dict) else None
        except Exception as e:
            logging.debug(f"[{product_id}] GraphQL attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                await asyncio.sleep(backoff)
                backoff *= 2
    logging.error(f"[{product_id}] GraphQL request failed after {max_retries} retries")
    return None


async def fetch_first_review_date(session, product_id: str, review_count: int,
                                  rate_limiter: Optional[AdaptiveRateLimiter] = None) -> Tuple[Optional[str], str]:
    """
    Find the date of a product's earliest review.

    Returns (date, status):
      ("YYYY-MM-DD", "ok")        earliest review found
      (None, "no_reviews")        API returned no evaluations at all
      (None, "fetch_failed")      network / API failure; safe to retry later
    """
    offset = max(int(review_count or 0) - EVALUATIONS_PAGE_SIZE, 0)

    # Jump to the tail. If number_of_ratings overshoots the API's count the jump
    # lands past the end and returns no rows, so back up a page at a time.
    page = None
    for _ in range(EVALUATIONS_MAX_BACKSTEPS + 1):
        page = await fetch_evaluations_page(session, product_id, encode_evaluations_cursor(offset), rate_limiter)
        if page is None:
            return None, "fetch_failed"
        if page.get("edges"):
            break
        if offset == 0:
            return None, "no_reviews"
        offset = max(offset - EVALUATIONS_PAGE_SIZE, 0)
    else:
        return None, "fetch_failed"

    dates = [e["node"]["createdAt"] for e in page["edges"]
             if isinstance(e.get("node"), dict) and e["node"].get("createdAt")]

    # If reviews were added after the metadata scrape the tail extends further;
    # keep following lastCursor until the API says there is no next page.
    pages = 1
    while page.get("pageInfo", {}).get("hasNextPage") and pages < EVALUATIONS_MAX_FOLLOW_PAGES:
        cursor = page["pageInfo"].get("lastCursor")
        if not cursor:
            break
        page = await fetch_evaluations_page(session, product_id, cursor, rate_limiter)
        if page is None:
            return None, "fetch_failed"
        if not page.get("edges"):
            break
        dates += [e["node"]["createdAt"] for e in page["edges"]
                  if isinstance(e.get("node"), dict) and e["node"].get("createdAt")]
        pages += 1

    if not dates:
        return None, "no_reviews"
    return min(dates)[:10], "ok"


async def deep_scrape_products(config_name: str, db_file: str, limit: Optional[int] = None,
                               concurrent: int = 4, force: bool = False):
    """
    Fill product_metadata.first_review_date for every product with ratings by
    querying the evaluations GraphQL API directly (no browser needed).

    Skips rows that already have a first_review_date (a real date or the
    'no_reviews' sentinel) unless force=True. Rows that failed are left NULL so
    the next run retries them.
    """
    logging.info(f"=" * 60)
    logging.info(f"DEEP SCRAPE WORKFLOW (first review date) - Configuration: {config_name}")
    if force:
        logging.info("MODE: Recomputing ALL products")
    logging.info(f"=" * 60)

    async with aiosqlite.connect(db_file) as db:
        if force:
            await db.execute("UPDATE product_metadata SET first_review_date = NULL")
        # Products with 0 ratings never get a review date; mark them without a request
        cursor = await db.execute("""
            UPDATE product_metadata
            SET first_review_date = 'no_reviews'
            WHERE first_review_date IS NULL
              AND (number_of_ratings IS NULL OR CAST(number_of_ratings AS INTEGER) = 0)
        """)
        bulk_skipped = cursor.rowcount
        await db.commit()
        query = """
            SELECT url, number_of_ratings FROM product_metadata
            WHERE first_review_date IS NULL
              AND CAST(number_of_ratings AS INTEGER) > 0
            ORDER BY CAST(number_of_ratings AS INTEGER) DESC
        """
        if limit:
            query += f" LIMIT {int(limit)}"
        async with db.execute(query) as cursor:
            rows = await cursor.fetchall()
    if bulk_skipped:
        logging.info(f"Bulk-marked {bulk_skipped:,} products with 0 ratings as 'no_reviews'")

    if not rows:
        logging.info("All rated products already have first_review_date!")
        return

    logging.info(f"Products to query: {len(rows):,} (only those with ratings)")

    rate_limiter = AdaptiveRateLimiter(initial_delay=0.5, max_delay=30.0)
    semaphore = asyncio.Semaphore(concurrent)
    connector = aiohttp.TCPConnector(limit=concurrent + 5, limit_per_host=concurrent, keepalive_timeout=30)
    timeout = aiohttp.ClientTimeout(total=60, connect=10)

    found_count = no_reviews_count = failed_count = 0
    start_time = time.time()
    total_items = len(rows)
    batch_size = 50

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:

        async def lookup(url, number_of_ratings):
            async with semaphore:
                product_id = extract_product_id(url)
                if not product_id:
                    logging.warning(f"Could not parse product id from {url}")
                    return url, None, "fetch_failed"
                try:
                    count = int(str(number_of_ratings).replace(",", ""))
                except (TypeError, ValueError):
                    count = 0
                date, status = await fetch_first_review_date(session, product_id, count, rate_limiter)
                logging.info(f"[{product_id}] ratings={count} first_review={date or status}")
                return url, date, status

        for i in range(0, total_items, batch_size):
            batch = rows[i:i + batch_size]
            results = await asyncio.gather(*(lookup(url, n) for url, n in batch), return_exceptions=True)

            async with aiosqlite.connect(db_file) as db:
                for result in results:
                    if isinstance(result, Exception):
                        logging.error(f"Batch error: {result}")
                        failed_count += 1
                        continue
                    url, date, status = result
                    if status == "ok":
                        await db.execute("UPDATE product_metadata SET first_review_date = ? WHERE url = ?", (date, url))
                        found_count += 1
                    elif status == "no_reviews":
                        await db.execute("UPDATE product_metadata SET first_review_date = 'no_reviews' WHERE url = ?", (url,))
                        no_reviews_count += 1
                    else:
                        failed_count += 1  # left NULL so a later run retries it
                await db.commit()

            processed = min(i + batch_size, total_items)
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta_seconds = (total_items - processed) / rate if rate > 0 else 0
            logging.info(
                f"Progress: {processed:,}/{total_items:,} ({processed*100//total_items}%) | "
                f"Found: {found_count:,} | No reviews: {no_reviews_count:,} | Failed: {failed_count:,} | "
                f"Elapsed: {int(elapsed // 60)}m {int(elapsed % 60)}s | ETA: {int(eta_seconds // 60)}m {int(eta_seconds % 60)}s | "
                f"Rate: {rate:.1f}/s"
            )

    total_elapsed = time.time() - start_time
    logging.info(f"=" * 60)
    logging.info(f"DEEP SCRAPE COMPLETE")
    logging.info(f"Found first review dates for {found_count:,} products ({failed_count:,} failed, will retry next run)")
    logging.info(f"Total time: {int(total_elapsed // 60)}m {int(total_elapsed % 60)}s")
    logging.info(f"=" * 60)

###########################
# 3. DOWNLOAD FREE FILES  #
###########################

def product_id_or_hash(url: str) -> str:
    """Product ID for use in file names; falls back to a URL hash when the URL has no ID."""
    return extract_product_id(url) or str(abs(hash(url)) % (10 ** 8))

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
    product_id = product_id_or_hash(product_url)
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
            # Download did not auto-start - inspect the landing page
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
                                auto_extract: bool = True,
                                top_n: Optional[int] = None):
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
        product_id = product_id_or_hash(url)
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
                    f"Session expired - stopping ({remaining_count} products not downloaded). "
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
        print(f"[OK] Created configuration: {args.name}")
        print(f"   Database: {manager.get_db_file(args.name)}")
        print(f"   Edit: configs/{args.name}.json")
    else:
        print("[FAIL] Failed to create configuration")

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
        backfill=args.backfill,
    )

async def cmd_deepscrape(args):
    """Run deep scrape workflow to find first review dates."""
    manager = ConfigManager()
    config = manager.get_config(args.config)
    if not config:
        logging.error(f"Configuration '{args.config}' not found")
        return
    
    db_file = manager.get_db_file(args.config)
    await setup_db(db_file)
    await deep_scrape_products(args.config, db_file, limit=args.limit,
                               concurrent=args.concurrent, force=args.force)

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

        # Overlap between search combinations (search_hits exists only on DBs created/migrated by setup_db)
        try:
            async with db.execute("SELECT COUNT(*), COUNT(DISTINCT product_id) FROM search_hits") as cursor:
                total_hits, hit_products = await cursor.fetchone()
            async with db.execute("""
                SELECT SUM(n_hits = 1), SUM(n_hits BETWEEN 2 AND 5), SUM(n_hits > 5)
                FROM product_search_summary
            """) as cursor:
                hits_1, hits_2_5, hits_6p = [v or 0 for v in await cursor.fetchone()]
        except Exception:
            total_hits = hit_products = hits_1 = hits_2_5 = hits_6p = 0
        
        # Metadata scraped
        async with db.execute("SELECT COUNT(*) FROM product_metadata") as cursor:
            metadata_count = (await cursor.fetchone())[0]

        # Apollo-state / first-review coverage (columns added by migration)
        async with db.execute("SELECT COUNT(*) FROM product_metadata WHERE post_date IS NOT NULL") as cursor:
            post_date_count = (await cursor.fetchone())[0]
        async with db.execute("""
            SELECT
              SUM(CASE WHEN first_review_date NOT IN ('no_reviews') AND first_review_date IS NOT NULL THEN 1 ELSE 0 END),
              SUM(CASE WHEN first_review_date IS NULL AND CAST(number_of_ratings AS INTEGER) > 0 THEN 1 ELSE 0 END)
            FROM product_metadata
        """) as cursor:
            row = await cursor.fetchone()
            first_review_count = row[0] or 0
            first_review_pending = row[1] or 0
        
        # Downloads - failure_reason column added by migration; may not exist on older DBs
        async with db.execute("SELECT COUNT(*), SUM(file_size) FROM downloads WHERE file_path IS NOT NULL") as cursor:
            row = await cursor.fetchone()
            downloads_count = row[0]
            total_size = row[1] or 0

        try:
            async with db.execute("SELECT COUNT(*) FROM downloads WHERE failure_reason IS NOT NULL") as cursor:
                failed_downloads = (await cursor.fetchone())[0]
        except Exception:
            failed_downloads = 0
        
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
    if total_hits:
        print(f"  search hits (all appearances): {total_hits:,} across {hit_products:,} products")
        print(f"  seen in 1 combo: {hits_1:,} | 2-5 combos: {hits_2_5:,} | 6+ combos: {hits_6p:,}  (see view product_search_summary)")
    print(f"Metadata scraped:         {metadata_count:,}")
    print(f"  with post_date:         {post_date_count:,}  (run 'scrape --backfill' to fill)")
    print(f"  with first_review_date: {first_review_count:,}  ({first_review_pending:,} rated products pending 'deepscrape')")
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

  # Re-fetch existing products that predate the post_date / Apollo-state fields
  python tpt_scraper_refactored.py scrape classroom-mgmt --backfill
  
  # Deep scrape to find first review dates (GraphQL, no browser)
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
    scrape_parser.add_argument("--backfill", action="store_true",
                               help="Re-scrape only existing products missing post_date (resumable)")
    scrape_parser.add_argument("--no-cache", action="store_true", dest="no_cache", help="Bypass HTTP cache for fresh fetches")
    
    # Deep scrape (evaluations GraphQL API)
    deepscrape_parser = subparsers.add_parser("deepscrape", help="Find each rated product's first review date via the evaluations API")
    deepscrape_parser.add_argument("config", help="Configuration name")
    deepscrape_parser.add_argument("--limit", type=int, help="Limit number of products to query")
    deepscrape_parser.add_argument("--concurrent", type=int, default=4, help="Concurrent API requests (default: 4)")
    deepscrape_parser.add_argument("--force", action="store_true", help="Recompute first_review_date for ALL products")
    
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
