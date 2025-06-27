import asyncio
import aiohttp
import aiohttp_client_cache
import async_timeout
import random
import json
import logging
import aiosqlite
import math
import os
import re
import argparse
import time
from pathlib import Path
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from tqdm.asyncio import tqdm_asyncio
from collections import deque
from typing import Optional, Tuple

CONFIG_FILE = "config.json"
DB_FILE = "scrape_cache.db"

# Load configuration

def load_config(config_file=CONFIG_FILE):
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.error("Error loading config: %s", e)
        return {
            "resource_type": ["", "teacher-tools"],
            "grade_level": ["", "elementary"],
            "subject": ["social-emotional"],
            "format": ["", "pdf"],
            "price_options": ["", "free"],
            "supports": [""],
            "sorting_methods": ["Relevance", "Rating"],
            "total_pages": 42,
            "concurrent_requests": 10,
            "max_retries": 3,
            "sleep_between_batches": [2, 5],
            "download_sleep": [5, 10],
            "retry_backoff_start": 1,
            "retry_backoff_factor": 2
        }

config = load_config()
RESOURCE_TYPES = config.get("resource_type", [])
GRADE_LEVELS = config.get("grade_level", [])
SUBJECTS = config.get("subject", [])
FORMATS = config.get("format", [])
PRICE_OPTIONS = config.get("price_options", [])
SUPPORTS = config.get("supports", [])
SORTING_METHODS = config.get("sorting_methods", [])

class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts based on response times and errors."""
    
    def __init__(self, initial_delay=1.0, max_delay=30.0, error_threshold=0.1):
        self.current_delay = initial_delay
        self.max_delay = max_delay
        self.error_threshold = error_threshold
        self.recent_responses = deque(maxlen=100)  # Track last 100 responses
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
            
        # Calculate recent error rate
        recent_errors = sum(1 for success, _, _ in self.recent_responses if not success)
        error_rate = recent_errors / len(self.recent_responses)
        
        # Calculate average response time
        avg_response_time = sum(rt for _, rt, _ in self.recent_responses) / len(self.recent_responses)
        
        # Adjust delay based on error rate and response time
        if error_rate > self.error_threshold:
            # Increase delay if too many errors
            self.current_delay = min(self.current_delay * 1.5, self.max_delay)
        elif error_rate < self.error_threshold / 2 and avg_response_time < 2.0:
            # Decrease delay if doing well
            self.current_delay = max(self.current_delay * 0.9, 0.1)
            
        return self.current_delay
        
    async def wait(self):
        """Wait for the current delay period."""
        delay = self.adjust_delay()
        await asyncio.sleep(delay)

class SmartBatcher:
    """Smart batcher that optimizes batch sizes based on performance."""
    
    def __init__(self, min_batch_size=5, max_batch_size=50, target_time=30.0):
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_time = target_time
        self.current_batch_size = min_batch_size
        self.performance_history = deque(maxlen=10)
        
    def record_batch_performance(self, batch_size: int, duration: float, success_rate: float):
        """Record performance of a batch."""
        self.performance_history.append((batch_size, duration, success_rate))
        
    def get_next_batch_size(self) -> int:
        """Calculate optimal batch size based on performance history."""
        if len(self.performance_history) < 3:
            return self.current_batch_size
            
        # Analyze recent performance
        recent_performance = list(self.performance_history)[-3:]
        avg_duration = sum(d for _, d, _ in recent_performance) / len(recent_performance)
        avg_success_rate = sum(sr for _, _, sr in recent_performance) / len(recent_performance)
        
        # Adjust batch size based on performance
        if avg_success_rate < 0.8:  # Too many failures
            self.current_batch_size = max(self.current_batch_size - 5, self.min_batch_size)
        elif avg_duration < self.target_time * 0.7 and avg_success_rate > 0.95:
            # Can handle more
            self.current_batch_size = min(self.current_batch_size + 3, self.max_batch_size)
        elif avg_duration > self.target_time * 1.3:
            # Taking too long
            self.current_batch_size = max(self.current_batch_size - 2, self.min_batch_size)
            
        return self.current_batch_size
TOTAL_PAGES = config.get("total_pages", 42)
CONCURRENT_REQUESTS = config.get("concurrent_requests", 10)
MAX_RETRIES = config.get("max_retries", 3)
SLEEP_BETWEEN_BATCHES = config.get("sleep_between_batches", [2, 5])
DOWNLOAD_SLEEP = config.get("download_sleep", [5, 10])
RETRY_BACKOFF_START = config.get("retry_backoff_start", 1)
RETRY_BACKOFF_FACTOR = config.get("retry_backoff_factor", 2)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

###########################
# SQLite Database Helpers #
###########################

async def setup_db():
    """Creates the SQLite database and tables if they don't exist."""
    async with aiosqlite.connect(DB_FILE) as db:
        # Enhanced SQLite performance settings
        await db.execute("PRAGMA journal_mode=WAL;")
        await db.execute("PRAGMA synchronous=NORMAL;")
        await db.execute("PRAGMA cache_size=10000;")  # 10MB cache
        await db.execute("PRAGMA temp_store=MEMORY;")
        await db.execute("PRAGMA mmap_size=268435456;")  # 256MB memory mapping
        await db.execute("PRAGMA optimize;")

        await db.execute("""
            CREATE TABLE IF NOT EXISTS extracted_urls (
                url TEXT PRIMARY KEY,
                resource_type TEXT,
                grade_level TEXT,
                subject TEXT,
                format TEXT,
                price_option TEXT,
                supports TEXT,
                sort_order TEXT,
                page INTEGER
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS extracted_pages (
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
        # Extraction progress table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS extraction_progress (
                id INTEGER PRIMARY KEY CHECK(id=1),
                last_batch INTEGER
            )
        """)
        await db.execute("""
            CREATE VIEW IF NOT EXISTS unique_extracted_urls AS
            SELECT url,
                   MIN(resource_type) AS resource_type,
                   MIN(grade_level) AS grade_level,
                   MIN(subject) AS subject,
                   MIN(format) AS format,
                   MIN(price_option) AS price_option,
                   MIN(supports) AS supports,
                   MIN(sort_order) AS sort_order,
                   MIN(page) AS page
            FROM extracted_urls
            GROUP BY url
        """)
        # Updated product_data table with new URL parameters.
        await db.execute("""
            CREATE TABLE IF NOT EXISTS product_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                short_description TEXT,
                long_description TEXT,
                rating_value TEXT,
                number_of_ratings TEXT,
                product_price TEXT,
                preview_keywords TEXT,
                url TEXT UNIQUE,
                resource_type TEXT,
                grade_level TEXT,
                subject TEXT,
                format TEXT,
                price_option TEXT,
                supports TEXT,
                sort_order TEXT,
                page INTEGER,
                config_metadata TEXT
            )
        """)
        
        # --- Add this migration code ---
        async with db.execute("PRAGMA table_info(product_data)") as cursor:
            columns = await cursor.fetchall()
        if not any(col[1] == "preview_keywords" for col in columns):
            await db.execute("ALTER TABLE product_data ADD COLUMN preview_keywords TEXT")
            await db.commit()
        # --- End migration code ---
        
        # Create performance indexes
        await db.execute("CREATE INDEX IF NOT EXISTS idx_extracted_urls_url ON extracted_urls(url);")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_product_data_url ON product_data(url);")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_product_data_price_option ON product_data(price_option);")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_product_data_resource_type ON product_data(resource_type);")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_product_data_preview_keywords ON product_data(preview_keywords);")
        
        # Add performance monitoring table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS performance_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                stage TEXT,
                batch_size INTEGER,
                duration REAL,
                success_rate REAL,
                error_rate REAL,
                items_processed INTEGER
            )
        """)
        
        # Add empty combinations tracking table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS empty_combinations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resource_type TEXT,
                grade_level TEXT,
                subject TEXT,
                format TEXT,
                price_option TEXT,
                supports TEXT,
                sort_order TEXT,
                discovered_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(resource_type, grade_level, subject, format, price_option, supports, sort_order)
            )
        """)
        
        await db.commit()

async def insert_product_data(data):
    """
    Insert product data along with configuration parameters.
    Data should be a tuple in the form:
    (title, short_description, long_description, rating_value, number_of_ratings,
     product_price, preview_keywords, url, resource_type, grade_level, subject, format, price_option, supports, sort_order, page, config_metadata)
    """
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute("""
            INSERT OR IGNORE INTO product_data 
            (title, short_description, long_description, rating_value, number_of_ratings, product_price, preview_keywords, url, resource_type, grade_level, subject, format, price_option, supports, sort_order, page, config_metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, data)
        await db.commit()

async def get_unique_extracted_urls():
    async with aiosqlite.connect(DB_FILE) as db:
        async with db.execute(
            "SELECT url, resource_type, grade_level, subject, format, price_option, supports, sort_order, page FROM unique_extracted_urls"
        ) as cursor:
            return await cursor.fetchall()

async def get_empty_combinations():
    """Get list of combinations known to be empty."""
    async with aiosqlite.connect(DB_FILE) as db:
        async with db.execute("""
            SELECT resource_type, grade_level, subject, format, price_option, supports, sort_order
            FROM empty_combinations
        """) as cursor:
            return set(await cursor.fetchall())

async def mark_combination_empty(resource_type, grade_level, subject, format_type, price_option, supports, sort_order):
    """Mark a combination as empty to avoid future processing."""
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute("""
            INSERT OR IGNORE INTO empty_combinations 
            (resource_type, grade_level, subject, format, price_option, supports, sort_order)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (resource_type, grade_level, subject, format_type, price_option, supports, sort_order))
        await db.commit()

async def load_valid_combinations():
    """Load pre-filtered valid combinations if available."""
    try:
        with open("valid_combinations.json", "r") as f:
            data = json.load(f)
        valid_combos = [tuple(combo) for combo in data.get("valid_combinations", [])]
        logging.info(f"Loaded {len(valid_combos):,} pre-filtered valid combinations")
        return valid_combos
    except FileNotFoundError:
        logging.info("No pre-filtered combinations found, will process all combinations")
        return None

###########################
# URL Building and Fetch  #
###########################

def build_page_url(resource_type, grade_level, subject, format_type, price_option, supports, sort_order, page):
    """
    Builds a complete URL based on the new URL structure:
    https://www.teacherspayteachers.com/browse/[resource-type]/[grade-level]/[subject]/[format]/[price]/[supports]?order=[sorting-method]
    """
    url_parts = ["https://www.teacherspayteachers.com/browse"]
    
    # Add each URL component if it's not empty
    if resource_type:
        url_parts.append(resource_type)
    if grade_level:
        url_parts.append(grade_level)
    if subject:
        url_parts.append(subject)
    if format_type:
        url_parts.append(format_type)
    if price_option:
        url_parts.append(price_option)
    if supports:
        url_parts.append(supports)
    
    base_url = "/".join(url_parts)
    
    # Add query parameters
    query_params = []
    
    # Add sorting (unless it's Relevance which is default)
    if sort_order and sort_order != "Relevance":
        query_params.append(f"order={sort_order}")
    
    # Add page number if not the first page
    if page > 1:
        query_params.append(f"page={page}")
    
    if query_params:
        return f"{base_url}?{'&'.join(query_params)}"
    else:
        return base_url

async def fetch(session, url, rate_limiter: Optional[AdaptiveRateLimiter] = None):
    """Fetches a URL using aiohttp with adaptive rate limiting and improved retry logic."""
    retries = 0
    backoff = config.get("retry_backoff_start", 1)
    backoff_factor = config.get("retry_backoff_factor", 2)
    max_retries = config.get("max_retries", 3)
    
    while retries < max_retries:
        start_time = time.time()
        try:
            # Apply rate limiting
            if rate_limiter:
                await rate_limiter.wait()
                
            async with async_timeout.timeout(30):
                async with session.get(url) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        if rate_limiter:
                            rate_limiter.record_response(True, response_time)
                        return await response.text()
                    elif response.status == 429:  # Rate limited
                        if rate_limiter:
                            rate_limiter.record_response(False, response_time)
                        retry_after = response.headers.get('Retry-After', backoff)
                        try:
                            retry_after = int(retry_after)
                        except (ValueError, TypeError):
                            retry_after = backoff
                        logging.warning("Rate limited (429) for URL: %s. Waiting %s seconds", url, retry_after)
                        await asyncio.sleep(retry_after)
                        raise Exception(f"Rate limited (429)")
                    elif response.status >= 500:  # Server error - retry
                        if rate_limiter:
                            rate_limiter.record_response(False, response_time)
                        logging.warning("Server error %s for URL: %s", response.status, url)
                        raise Exception(f"Server error {response.status}")
                    else:  # Client error - don't retry
                        if rate_limiter:
                            rate_limiter.record_response(False, response_time)
                        logging.warning("Client error %s for URL: %s", response.status, url)
                        return None
                        
        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            if rate_limiter:
                rate_limiter.record_response(False, response_time)
            retries += 1
            logging.warning("Timeout fetching %s. Retry %s/%s in %s sec", url, retries, max_retries, backoff)
            if retries < max_retries:
                await asyncio.sleep(backoff)
                backoff *= backoff_factor
        except Exception as e:
            response_time = time.time() - start_time
            if rate_limiter:
                rate_limiter.record_response(False, response_time)
            retries += 1
            logging.warning("Error fetching %s: %s. Retry %s/%s in %s sec", url, e, retries, max_retries, backoff)
            if retries < max_retries:
                await asyncio.sleep(backoff)
                backoff *= backoff_factor
                
    logging.error("Failed to fetch %s after %s retries", url, max_retries)
    return None

###########################
# Extraction Stage        #
###########################

async def extract_urls_from_page(session, page_url, rate_limiter: Optional[AdaptiveRateLimiter] = None):
    """Extracts product URLs from a single page using BeautifulSoup (lxml parser)."""
    html = await fetch(session, page_url, rate_limiter)
    if not html:
        return []
    soup = BeautifulSoup(html, "lxml")
    product_elements = soup.select("a.ProductRowCard-module__cardTitleLink--YPqiC")
    urls = []
    for product in product_elements:
        href = product.get("href")
        if href:
            full_url = f"https://www.teacherspayteachers.com{href}" if not href.startswith("http") else href
            urls.append(full_url)
    return urls

async def extract_page_wrapper(session, page_url, semaphore, resource_type, grade_level, subject, format_type, price_option, supports, sort_order, page, rate_limiter: Optional[AdaptiveRateLimiter] = None):
    async with semaphore:
        logging.info("Fetching page: %s", page_url)
        urls = await extract_urls_from_page(session, page_url, rate_limiter)
        
        # Smart empty combination detection
        if page == 1 and len(urls) == 0:
            # If page 1 of a combination is empty, mark the whole combination as empty
            await mark_combination_empty(resource_type, grade_level, subject, format_type, price_option, supports, sort_order)
            logging.info(f"Marked empty combination: {resource_type}/{grade_level}/{subject}/{format_type}/{price_option}/{supports}/{sort_order}")
        
        if urls:
            async with aiosqlite.connect(DB_FILE) as db:
                # Immediately mark the page as processed and insert URLs
                await db.execute(
                    "INSERT OR IGNORE INTO extracted_pages VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (resource_type, grade_level, subject, format_type, price_option, supports, sort_order, page),
                )
                for url in urls:
                    await db.execute(
                        "INSERT OR IGNORE INTO extracted_urls (url, resource_type, grade_level, subject, format, price_option, supports, sort_order, page) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (url, resource_type, grade_level, subject, format_type, price_option, supports, sort_order, page)
                    )
                await db.commit()
        else:
            # Even if no URLs, mark the page as processed to avoid re-processing
            async with aiosqlite.connect(DB_FILE) as db:
                await db.execute(
                    "INSERT OR IGNORE INTO extracted_pages VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (resource_type, grade_level, subject, format_type, price_option, supports, sort_order, page),
                )
                await db.commit()
        
        return urls

async def extraction_stage(initial_batch_size=None):
    logging.info("Using SQLite database at %s", os.path.abspath(DB_FILE))

    # Initialize adaptive components
    rate_limiter = AdaptiveRateLimiter(
        initial_delay=config.get("sleep_between_batches", [0.5, 2.0])[0],
        max_delay=30.0,
        error_threshold=0.15
    )
    
    batcher = SmartBatcher(
        min_batch_size=config.get("min_batch_size", 10),
        max_batch_size=config.get("max_batch_size", 50),
        target_time=config.get("target_batch_time", 30.0)
    )
    
    if initial_batch_size:
        batcher.current_batch_size = initial_batch_size

    # 1️⃣ Check for pre-filtered combinations first
    valid_base_combos = await load_valid_combinations()
    
    if valid_base_combos:
        logging.info(f"Using pre-filtered combinations: {len(valid_base_combos):,}")
        # Generate combinations from pre-filtered list
        combos = [
            (*combo, page)
            for combo in valid_base_combos
            for page in range(1, TOTAL_PAGES + 1)
        ]
    else:
        # 2️⃣ Generate all combinations and filter out known empty ones
        all_base_combos = [
            (resource_type, grade_level, subject, format_type, price_option, supports, sort_order)
            for resource_type in RESOURCE_TYPES
            for grade_level in GRADE_LEVELS
            for subject in SUBJECTS
            for format_type in FORMATS
            for price_option in PRICE_OPTIONS
            for supports in SUPPORTS
            for sort_order in SORTING_METHODS
        ]
        
        # Load known empty combinations
        empty_combos = await get_empty_combinations()
        logging.info(f"Found {len(empty_combos):,} known empty combinations to skip")
        
        # Filter out empty combinations
        valid_base_combos = [combo for combo in all_base_combos if combo not in empty_combos]
        filtered_out = len(all_base_combos) - len(valid_base_combos)
        
        logging.info(f"Filtered out {filtered_out:,} empty combinations ({filtered_out/len(all_base_combos)*100:.1f}%)")
        
        # Generate full combinations with pages
        combos = [
            (*combo, page)
            for combo in valid_base_combos
            for page in range(1, TOTAL_PAGES + 1)
        ]

    # 3️⃣ Load already‑done combos from the DB
    async with aiosqlite.connect(DB_FILE) as db:
        async with db.execute("""
            SELECT resource_type, grade_level, subject, format, price_option, supports, sort_order, page 
            FROM extracted_pages
        """) as cursor:
            done_combos = set(await cursor.fetchall())

    # 4️⃣ Filter out completed combos
    remaining = [c for c in combos if c not in done_combos]
    
    # Enhanced progress display
    completed_count = len(done_combos)
    total_count = len(combos)
    remaining_count = len(remaining)
    completion_percentage = (completed_count / total_count * 100) if total_count > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"EXTRACTION PROGRESS SUMMARY")
    print(f"{'='*60}")
    print(f"Total combinations:     {total_count:,}")
    print(f"Already completed:      {completed_count:,} ({completion_percentage:.1f}%)")
    print(f"Remaining to process:   {remaining_count:,}")
    if remaining_count > 0:
        estimated_time = remaining_count * 2 / 3600  # Rough estimate: 2 seconds per combination
        print(f"Estimated time:         {estimated_time:.1f} hours")
    print(f"{'='*60}\n")
    
    if not remaining:
        logging.info("🎉 All combinations already processed!")
        return
    
    semaphore = asyncio.Semaphore(config.get("concurrent_requests", 25))

    # Enhanced session configuration with connection pooling
    connector = aiohttp.TCPConnector(
        limit=config.get("concurrent_requests", 25) + 10,  # Slightly higher than concurrent requests
        limit_per_host=config.get("concurrent_requests", 25),
        keepalive_timeout=30,
        enable_cleanup_closed=True
    )
    
    timeout = aiohttp.ClientTimeout(total=60, connect=10)
    
    async with aiohttp_client_cache.CachedSession(
        cache_name="aiohttp_cache", 
        expire_after=3600,
        connector=connector,
        timeout=timeout
    ) as session:
        
        batch_idx = 0
        processed = 0
        
        while processed < len(remaining):
            batch_size = batcher.get_next_batch_size()
            start = processed
            end = min(processed + batch_size, len(remaining))
            batch = remaining[start:end]
            
            logging.info(f"Processing batch {batch_idx + 1} with {len(batch)} items (adaptive batch size: {batch_size})")
            
            batch_start_time = time.time()
            tasks = [
                extract_page_wrapper(session,
                                     build_page_url(resource_type, grade_level, subject, format_type, price_option, supports, sort_order, page),
                                     semaphore,
                                     resource_type, grade_level, subject, format_type, price_option, supports, sort_order, page,
                                     rate_limiter)
                for resource_type, grade_level, subject, format_type, price_option, supports, sort_order, page in batch
            ]

            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                batch_duration = time.time() - batch_start_time
                
                # Calculate success rate
                successes = sum(1 for r in results if not isinstance(r, Exception))
                success_rate = successes / len(results) if results else 0
                error_rate = 1.0 - success_rate
                
                # Record batch performance for adaptive batching
                batcher.record_batch_performance(batch_size, batch_duration, success_rate)
                
                # Record performance stats to database
                await record_performance_stats("extraction", batch_size, batch_duration, success_rate, error_rate, len(batch))
                
                # Enhanced progress logging
                overall_progress = (processed / len(remaining)) * 100
                estimated_remaining_time = ((len(remaining) - processed) * batch_duration / len(batch)) / 3600 if len(batch) > 0 else 0
                
                logging.info(f"Batch {batch_idx + 1} complete in {batch_duration:.1f}s. Success rate: {success_rate:.2%}")
                logging.info(f"Progress: {processed:,}/{len(remaining):,} ({overall_progress:.1f}%) | "
                           f"ETA: {estimated_remaining_time:.1f}h | Batch size: {batch_size}")
                
            except Exception as e:
                logging.error("Error processing batch %s: %s", batch_idx + 1, e)
                # Record poor performance for this batch
                batch_duration = time.time() - batch_start_time
                batcher.record_batch_performance(batch_size, batch_duration, 0.0)
                await record_performance_stats("extraction", batch_size, batch_duration, 0.0, 1.0, len(batch))

            processed = end
            batch_idx += 1
            
            # Adaptive sleep between batches
            if processed < len(remaining):
                sleep_range = config.get("sleep_between_batches", [0.5, 2.0])
                sleep_time = random.uniform(sleep_range[0], sleep_range[1])
                await asyncio.sleep(sleep_time)

    logging.info("Adaptive extraction finished; all new pages saved.")
    logging.info(f"Final batch size: {batcher.current_batch_size}, Final delay: {rate_limiter.current_delay:.2f}s")
    logging.info(f"Total requests: {rate_limiter.total_count}, Error rate: {rate_limiter.error_count / max(rate_limiter.total_count, 1):.2%}")
    
    # Show empty combination stats
    empty_combos = await get_empty_combinations()
    logging.info(f"Total empty combinations discovered: {len(empty_combos):,}")

###########################
# Extraction Test Stage   #
###########################

async def extraction_test(test_limit=5):
    """
    Test extraction stage that fetches a single page from a single combination,
    stores up to test_limit URLs with their parameters in the SQLite database,
    and prints them.
    """
    # Initialize rate limiter for testing
    rate_limiter = AdaptiveRateLimiter(initial_delay=0.5, max_delay=10.0)
    
    extracted_url_details = {}  # dict mapping url -> (resource_type, grade_level, subject, format_type, price_option, supports, sort_order, page)
    
    connector = aiohttp.TCPConnector(limit=10, keepalive_timeout=30)
    timeout = aiohttp.ClientTimeout(total=30, connect=10)
    
    async with aiohttp_client_cache.CachedSession(
        cache_name="aiohttp_cache", 
        expire_after=3600,
        connector=connector,
        timeout=timeout
    ) as session:
        resource_type = RESOURCE_TYPES[0] if RESOURCE_TYPES else ""
        grade_level = GRADE_LEVELS[0] if GRADE_LEVELS else ""
        subject = SUBJECTS[0] if SUBJECTS else "social-emotional"
        format_type = FORMATS[0] if FORMATS else ""
        price_option = PRICE_OPTIONS[0] if PRICE_OPTIONS else ""
        supports = SUPPORTS[0] if SUPPORTS else ""
        sort_order = SORTING_METHODS[0] if SORTING_METHODS else "Relevance"
        page = 1
        page_url = build_page_url(resource_type, grade_level, subject, format_type, price_option, supports, sort_order, page)
        logging.info("Test fetching page: %s", page_url)
        urls = await extract_urls_from_page(session, page_url, rate_limiter)
        for url in urls:
            if len(extracted_url_details) >= test_limit:
                break
            extracted_url_details[url] = (resource_type, grade_level, subject, format_type, price_option, supports, sort_order, page)
    logging.info("Test extracted %s URLs", len(extracted_url_details))
    async with aiosqlite.connect(DB_FILE) as db:
        for url, params in extracted_url_details.items():
            resource_type, grade_level, subject, format_type, price_option, supports, sort_order, page = params
            await db.execute("""
                INSERT OR IGNORE INTO extracted_urls (url, resource_type, grade_level, subject, format, price_option, supports, sort_order, page)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (url, resource_type, grade_level, subject, format_type, price_option, supports, sort_order, page))
        await db.commit()
    logging.info("Test extracted URLs saved to SQLite database (%s).", DB_FILE)
    for url, params in extracted_url_details.items():
        print(f"{url} -> {params}")
    
    # Print rate limiter stats
    logging.info(f"Rate limiter stats - Final delay: {rate_limiter.current_delay:.2f}s, "
                f"Total requests: {rate_limiter.total_count}, "
                f"Error rate: {rate_limiter.error_count / max(rate_limiter.total_count, 1):.2%}")

###########################
# Processing Stage        #
###########################

async def scrape_product_data(session, url, rate_limiter: Optional[AdaptiveRateLimiter] = None):
    """Scrapes product data from a product URL using BeautifulSoup (lxml parser)."""
    html = await fetch(session, url, rate_limiter)
    if not html:
        return None
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.string if soup.title else None
    meta_desc = soup.find("meta", {"name": "description"})
    short_description = meta_desc["content"] if meta_desc and meta_desc.has_attr("content") else None
    long_desc_elem = soup.find(class_="ProductDescriptionLayout__htmlDisplay")
    long_description = long_desc_elem.get_text(strip=True) if long_desc_elem else None
    rating_value = None
    number_of_ratings = None
    rating_elem = soup.select_one("span.StarRating-module__srOnly--FAzEA")
    if rating_elem:
        rating_text = rating_elem.get_text(separator=" ", strip=True)
        parts = rating_text.split()
        if len(parts) >= 3:
            rating_value = parts[1]
            number_of_ratings = parts[-2]
    # Try multiple methods to extract price
    product_price = None
    
    # Method 1: Check for meta tag with property="product:price:amount"
    meta_price = soup.find("meta", {"property": "product:price:amount"})
    if meta_price and meta_price.has_attr("content"):
        product_price = meta_price["content"]
    
    # Method 2: If not found, try JSON-LD structured data
    if not product_price:
        script_tags = soup.find_all("script", {"type": "application/ld+json"})
        for script in script_tags:
            try:
                data = json.loads(script.string)
                if isinstance(data, dict) and data.get("@type") == "Product":
                    offers = data.get("offers", {})
                    if offers and offers.get("price"):
                        product_price = offers.get("price")
                        break
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and item.get("@type") == "Product":
                            offers = item.get("offers", {})
                            if offers and offers.get("price"):
                                product_price = offers.get("price")
                                break
                    if product_price:
                        break
            except:
                continue
    
    # Method 3: If still not found, try CSS selectors for price elements
    if not product_price:
        price_selectors = [
            "[class*='Price'] span",
            "[class*='price']",
            "[data-testid*='price']"
        ]
        for selector in price_selectors:
            price_elem = soup.select_one(selector)
            if price_elem:
                price_text = price_elem.get_text(strip=True)
                # Extract price from text like "$120.00" or "FREE"
                if price_text and ('$' in price_text or 'free' in price_text.lower()):
                    if 'free' in price_text.lower():
                        product_price = "0.00"
                    else:
                        # Extract numeric price from text like "$120.00$180.00" -> "120.00"
                        price_match = re.search(r'\$([0-9]+\.?[0-9]*)', price_text)
                        if price_match:
                            product_price = price_match.group(1)
                    break

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

    return title, short_description, long_description, rating_value, number_of_ratings, product_price, preview_keywords, url

async def processing_stage(initial_batch_size=None):
    url_records = await get_unique_extracted_urls()  # Each record is (url, resource_type, grade_level, subject, format, price_option, supports, sort_order, page)
    total_records = len(url_records)
    logging.info("Total unique URLs to process: %s", total_records)

    # Initialize adaptive components
    rate_limiter = AdaptiveRateLimiter(
        initial_delay=config.get("download_sleep", [1.0, 3.0])[0],
        max_delay=30.0,
        error_threshold=0.15
    )
    
    batcher = SmartBatcher(
        min_batch_size=config.get("min_batch_size", 10),
        max_batch_size=config.get("max_batch_size", 50),
        target_time=config.get("target_batch_time", 45.0)  # Longer target for processing
    )
    
    if initial_batch_size:
        batcher.current_batch_size = initial_batch_size

    # Enhanced session configuration
    connector = aiohttp.TCPConnector(
        limit=config.get("concurrent_requests", 25) + 10,
        limit_per_host=config.get("concurrent_requests", 25),
        keepalive_timeout=30,
        enable_cleanup_closed=True
    )
    
    timeout = aiohttp.ClientTimeout(total=60, connect=10)
    
    async with aiohttp_client_cache.CachedSession(
        cache_name="aiohttp_cache", 
        expire_after=3600,
        connector=connector,
        timeout=timeout
    ) as session:
        
        processed = 0
        batch_idx = 0
        
        while processed < total_records:
            batch_size = batcher.get_next_batch_size()
            start = processed
            end = min(processed + batch_size, total_records)
            batch = url_records[start:end]
            batch_urls = [record[0] for record in batch]

            # Pre-check: query the DB for URLs already processed in this batch.
            async with aiosqlite.connect(DB_FILE) as db:
                placeholders = ','.join('?' for _ in batch_urls)
                query = f"SELECT url FROM product_data WHERE url IN ({placeholders})"
                async with db.execute(query, batch_urls) as cursor:
                    processed_rows = await cursor.fetchall()
                processed_urls = set(row[0] for row in processed_rows)

            # Only process URLs that haven't been scraped yet.
            tasks = []
            indices_to_process = []  # Keep track of which indices in the batch need scraping.
            for idx, record in enumerate(batch):
                url = record[0]
                if url not in processed_urls:
                    tasks.append(scrape_product_data(session, url, rate_limiter))
                    indices_to_process.append(idx)

            pre_skipped = len(batch) - len(tasks)  # Count of URLs pre-skipped from the batch.
            
            if tasks:
                logging.info(f"Processing batch {batch_idx + 1} with {len(tasks)} new URLs (batch size: {batch_size}, {pre_skipped} already processed)")
                batch_start_time = time.time()
                
                # Use semaphore to limit concurrent processing requests
                semaphore = asyncio.Semaphore(config.get("concurrent_requests", 25))
                
                async def process_with_sem(task):
                    async with semaphore:
                        return await task
                
                tasks_with_sem = [process_with_sem(task) for task in tasks]
                results = await asyncio.gather(*tasks_with_sem, return_exceptions=True)
                
                batch_duration = time.time() - batch_start_time
                
                # Calculate success rate
                successes = sum(1 for r in results if r and not isinstance(r, Exception))
                success_rate = successes / len(results) if results else 0
                error_rate = 1.0 - success_rate
                
                # Record batch performance
                batcher.record_batch_performance(batch_size, batch_duration, success_rate)
                
                # Record performance stats to database
                await record_performance_stats("processing", batch_size, batch_duration, success_rate, error_rate, len(tasks))
                
            else:
                results = []
                batch_duration = 0.1
                success_rate = 1.0  # All were pre-skipped, which is "success"
                error_rate = 0.0
                logging.info(f"Batch {batch_idx + 1}: All {len(batch)} URLs already processed")
                # Record skipped batch stats
                await record_performance_stats("processing", batch_size, batch_duration, success_rate, error_rate, 0)
                
            new_count = 0
            skipped_count = pre_skipped  # Start with pre-skipped URLs as skipped.

            # Insert newly scraped data into the DB.
            async with aiosqlite.connect(DB_FILE) as db:
                for result, idx in zip(results, indices_to_process):
                    if result and not isinstance(result, Exception):
                        # Retrieve associated parameters from the batch record.
                        url, resource_type, grade_level, subject, format_type, price_option, supports, sort_order, page = batch[idx]
                        # Build a configuration metadata JSON string from the extraction parameters.
                        config_metadata = json.dumps({
                            "resource_type": resource_type,
                            "grade_level": grade_level,
                            "subject": subject,
                            "format": format_type,
                            "price_option": price_option,
                            "supports": supports,
                            "sort_order": sort_order,
                            "page": page
                        })
                        full_data = (*result, resource_type, grade_level, subject, format_type, price_option, supports, sort_order, page, config_metadata)
                        cursor = await db.execute(
                            """
                            INSERT OR IGNORE INTO product_data 
                            (title, short_description, long_description, rating_value, number_of_ratings, product_price, preview_keywords, url, resource_type, grade_level, subject, format, price_option, supports, sort_order, page, config_metadata)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, full_data
                        )
                        if cursor.rowcount > 0:
                            new_count += 1
                        else:
                            skipped_count += 1
                    else:
                        skipped_count += 1
                await db.commit()

            processed = end
            batch_idx += 1
            percent_complete = (processed / total_records) * 100
            logging.info(
                f"Batch {batch_idx} complete in {batch_duration:.1f}s: {new_count} new, {skipped_count} skipped. "
                f"Success rate: {success_rate:.2%}. {processed}/{total_records} ({percent_complete:.2f}%) complete."
            )

            # Adaptive sleep between batches if there was new data
            if new_count > 0 and processed < total_records:
                sleep_range = config.get("download_sleep", [1.0, 3.0])
                sleep_time = random.uniform(sleep_range[0], sleep_range[1])
                await asyncio.sleep(sleep_time)
                
    logging.info("Finished processing unique URLs with adaptive optimizations.")
    logging.info(f"Final batch size: {batcher.current_batch_size}, Final delay: {rate_limiter.current_delay:.2f}s")
    logging.info(f"Total requests: {rate_limiter.total_count}, Error rate: {rate_limiter.error_count / max(rate_limiter.total_count, 1):.2%}")

#############################
# Free File Download Stage  #
#############################

async def download_free_file(prod_id, url, session_file="tpt_storage.json", dry_run=False, max_retries=3):
    from pathlib import Path

    free_url = url.replace("/Product/", "/FreeDownload/")
    downloads_dir = Path("downloads")
    downloads_dir.mkdir(exist_ok=True)

    if dry_run:
        logging.info(f"[Dry Run] Would download: {free_url}")
        return True  # Pretend success

    for attempt in range(1, max_retries + 1):
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    storage_state=session_file,
                    accept_downloads=True
                )
                page = await context.new_page()

                logging.info(f"[{prod_id}] Attempt {attempt}: Navigating to {free_url}")
                async with page.expect_download() as download_info:
                    await page.goto(free_url)

                download = await download_info.value
                suggested_name = download.suggested_filename
                save_path = downloads_dir / f"{prod_id}_{suggested_name}"
                await download.save_as(save_path)

                # Verify file exists and is non-empty
                if not save_path.exists() or save_path.stat().st_size == 0:
                    raise Exception("Downloaded file is empty or missing.")

                # Informative logging for downloaded file extension
                logging.info(f"[{prod_id}] File type: {save_path.suffix}")

                # Store in DB
                async with aiosqlite.connect(DB_FILE) as db:
                    await db.execute(
                        "INSERT OR IGNORE INTO free_file_downloads (product_id, free_file_path) VALUES (?, ?)",
                        (prod_id, str(save_path))
                    )
                    await db.commit()

                logging.info(f"[{prod_id}] ✅ Downloaded: {save_path}")
                await browser.close()
                return True

        except Exception as e:
            logging.warning(f"[{prod_id}] ❌ Attempt {attempt} failed: {e}")
            if attempt == max_retries:
                logging.error(f"[{prod_id}] ❌ Giving up after {max_retries} attempts.")
            await asyncio.sleep(2)  # backoff

    return False


from tqdm.asyncio import tqdm_asyncio

async def processing_free_download_stage(dry_run=False):
    # Filters
    allowed_folders = [
        "teacher-tools/classroom-management/elementary/preschool",
        "teacher-tools/classroom-management/elementary/kindergarten",
        "teacher-tools/classroom-management/elementary/1st-grade",
        "teacher-tools/classroom-management/elementary/2nd-grade",
        "teacher-tools/classroom-management/elementary/3rd-grade",
        "teacher-tools/classroom-management/elementary/4th-grade",
        "teacher-tools/classroom-management/elementary/5th-grade",
        "teacher-tools/classroom-management/middle-school/6th-grade",
        "teacher-tools/classroom-management/middle-school/7th-grade",
        "teacher-tools/classroom-management/middle-school/8th-grade",
        "teacher-tools/classroom-management/high-school/9th-grade",
        "teacher-tools/classroom-management/high-school/10th-grade",
        "teacher-tools/classroom-management/high-school/11th-grade",
        "teacher-tools/classroom-management/high-school/12th-grade",
        "teacher-tools/classroom-management/not-grade-specific",
    ]
    allowed_price_option = "free"
    allowed_sort_orders = [
        "Relevance", "Rating", "Rating-Count", "Price-Asc", "Price-Desc", "Most-Recent"
    ]

    async with aiosqlite.connect(DB_FILE) as db:
        async with db.execute(
            f"""
            SELECT id, url, product_price FROM product_data 
            WHERE price_option = ?
              AND folder IN ({','.join(['?'] * len(allowed_folders))})
              AND sort_order IN ({','.join(['?'] * len(allowed_sort_orders))})
              AND id NOT IN (SELECT product_id FROM free_file_downloads)
            """,
            [allowed_price_option] + allowed_folders + allowed_sort_orders
        ) as cursor:
            rows = await cursor.fetchall()

    logging.info(f"🔎 Found {len(rows)} products to download.")
    failures = []

    # Use asyncio tasks with progress bar
    sem = asyncio.Semaphore(5)

    async def worker(row):
        async with sem:
            prod_id, url, _ = row
            success = await download_free_file(prod_id, url, dry_run=dry_run)
            if not success:
                failures.append(prod_id)

    await tqdm_asyncio.gather(*(worker(row) for row in rows), desc="Downloading")

    if failures:
        logging.warning(f"❌ Failed to download {len(failures)} product(s): {failures}")
    else:
        logging.info("✅ All downloads succeeded.")



#############################
# Update & Check Config Metadata Functions
#############################

async def add_config_metadata_column_if_needed():
    """
    Check if the 'config_metadata' column exists in product_data.
    If not, add the column.
    """
    async with aiosqlite.connect(DB_FILE) as db:
        async with db.execute("PRAGMA table_info(product_data)") as cursor:
            columns = await cursor.fetchall()
        if not any(col[1] == "config_metadata" for col in columns):
            await db.execute("ALTER TABLE product_data ADD COLUMN config_metadata TEXT")
            await db.commit()
            logging.info("Added config_metadata column to product_data table.")
        else:
            logging.info("config_metadata column already exists.")

async def update_config_metadata():
    """
    Update existing product_data records by setting the config_metadata field 
    based on the existing URL parameters.
    """
    # First, ensure the column exists.
    await add_config_metadata_column_if_needed()
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute("""
            UPDATE product_data
            SET config_metadata = json_object(
                'resource_type', resource_type,
                'grade_level', grade_level,
                'subject', subject,
                'format', format,
                'price_option', price_option,
                'supports', supports,
                'sort_order', sort_order,
                'page', page
            )
            WHERE config_metadata IS NULL
        """)
        await db.commit()
    logging.info("Updated config_metadata for existing records.")

async def check_config_metadata():
    """
    Check and print out a sample of records with the config_metadata field set.
    """
    async with aiosqlite.connect(DB_FILE) as db:
        async with db.execute("SELECT COUNT(*) FROM product_data WHERE config_metadata IS NOT NULL") as cursor:
            count = await cursor.fetchone()
            print("Records with config_metadata set:", count[0])
        
        async with db.execute("SELECT id, resource_type, grade_level, subject, format, price_option, supports, sort_order, page, config_metadata FROM product_data LIMIT 5") as cursor:
            rows = await cursor.fetchall()
            for row in rows:
                print(row)

###########################
# Backfill Preview Keywords Stage
###########################

async def backfill_preview_keywords(batch_size=50, concurrency=20):
    """
    Find product_data records missing preview_keywords, scrape, and update them.
    """
    async with aiosqlite.connect(DB_FILE) as db:
        async with db.execute(
            "SELECT id, url FROM product_data WHERE preview_keywords IS NULL OR preview_keywords = ''"
        ) as cursor:
            rows = await cursor.fetchall()

    if not rows:
        logging.info("No records missing preview_keywords.")
        return

    logging.info(f"Backfilling preview_keywords for {len(rows)} records.")

    # Initialize rate limiter for backfill
    rate_limiter = AdaptiveRateLimiter(
        initial_delay=1.0,
        max_delay=15.0,
        error_threshold=0.1
    )

    total = len(rows)
    sem = asyncio.Semaphore(concurrency)
    
    # Enhanced session configuration
    connector = aiohttp.TCPConnector(
        limit=concurrency + 5,
        limit_per_host=concurrency,
        keepalive_timeout=30,
        enable_cleanup_closed=True
    )
    
    timeout = aiohttp.ClientTimeout(total=60, connect=10)
    
    async with aiohttp_client_cache.CachedSession(
        cache_name="aiohttp_cache", 
        expire_after=3600,
        connector=connector,
        timeout=timeout
    ) as session:
        async with aiosqlite.connect(DB_FILE) as db:  # Reuse connection
            for i in range(0, total, batch_size):
                batch = rows[i:i+batch_size]
                async def scrape_with_sem(url):
                    async with sem:
                        return await scrape_product_data(session, url, rate_limiter)
                tasks = [scrape_with_sem(url) for _, url in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                updates = []
                for (row, result) in zip(batch, results):
                    prod_id, url = row
                    if result and not isinstance(result, Exception):
                        preview_keywords = result[6]  # 7th item in tuple
                        if preview_keywords:
                            updates.append((preview_keywords, prod_id))
                if updates:
                    await db.executemany(
                        "UPDATE product_data SET preview_keywords = ? WHERE id = ?",
                        updates
                    )
                    await db.commit()
                completed = min(i + batch_size, total)
                percent = (completed / total) * 100
                logging.info(f"Updated {len(updates)} records in batch {i//batch_size+1}. {completed}/{total} ({percent:.2f}%%) complete.")

    logging.info("Backfill of preview_keywords complete.")
    logging.info(f"Rate limiter stats - Final delay: {rate_limiter.current_delay:.2f}s, "
                f"Total requests: {rate_limiter.total_count}, "
                f"Error rate: {rate_limiter.error_count / max(rate_limiter.total_count, 1):.2%}")

###########################
# Performance Tracking    #
###########################

async def record_performance_stats(stage: str, batch_size: int, duration: float, success_rate: float, error_rate: float, items_processed: int):
    """Record performance statistics to the database for monitoring and optimization."""
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute("""
            INSERT INTO performance_stats (stage, batch_size, duration, success_rate, error_rate, items_processed)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (stage, batch_size, duration, success_rate, error_rate, items_processed))
        await db.commit()

async def get_performance_summary(stage: str = None, limit: int = 100) -> list:
    """Get recent performance statistics for analysis."""
    async with aiosqlite.connect(DB_FILE) as db:
        if stage:
            query = """
                SELECT timestamp, stage, batch_size, duration, success_rate, error_rate, items_processed
                FROM performance_stats 
                WHERE stage = ?
                ORDER BY timestamp DESC 
                LIMIT ?
            """
            params = (stage, limit)
        else:
            query = """
                SELECT timestamp, stage, batch_size, duration, success_rate, error_rate, items_processed
                FROM performance_stats 
                ORDER BY timestamp DESC 
                LIMIT ?
            """
            params = (limit,)
            
        async with db.execute(query, params) as cursor:
            return await cursor.fetchall()

async def show_performance_stats(stage: str = None, limit: int = 20):
    """Display performance statistics."""
    stats = await get_performance_summary(stage, limit)
    
    if not stats:
        print("No performance statistics available.")
        return
        
    print(f"\n{'='*80}")
    print(f"PERFORMANCE STATISTICS{f' - {stage.upper()}' if stage else ''}")
    print(f"{'='*80}")
    print(f"{'Timestamp':<20} {'Stage':<12} {'Batch':<6} {'Duration':<8} {'Success':<8} {'Error':<8} {'Items':<6}")
    print(f"{'-'*80}")
    
    for stat in stats:
        timestamp, stage_name, batch_size, duration, success_rate, error_rate, items = stat
        print(f"{timestamp[:19]:<20} {stage_name:<12} {batch_size:<6} {duration:<8.1f} {success_rate:<8.1%} {error_rate:<8.1%} {items:<6}")
    
    # Calculate averages
    if stats:
        avg_duration = sum(s[3] for s in stats) / len(stats)
        avg_success = sum(s[4] for s in stats) / len(stats)
        avg_error = sum(s[5] for s in stats) / len(stats)
        total_items = sum(s[6] for s in stats)
        
        print(f"{'-'*80}")
        print(f"{'AVERAGES':<20} {'':<12} {'':<6} {avg_duration:<8.1f} {avg_success:<8.1%} {avg_error:<8.1%} {total_items:<6}")
        print(f"{'='*80}\n")

###########################
# Main Entry Point        #
###########################

async def main():
    parser = argparse.ArgumentParser(
        description="TPT Scraper Tool: Extract, process, and download TPT product data with adaptive performance optimization."
    )
    subparsers = parser.add_subparsers(dest="stage", required=True, help="Stage to run")

    # Test stage
    subparsers.add_parser("test", help="Test extraction on a single page.")

    # Extract stage
    extract_parser = subparsers.add_parser("extract", help="Extract product URLs for all config combinations.")
    extract_parser.add_argument("--batch-size", type=int, help="Initial batch size (will be optimized adaptively)")

    # Process stage
    process_parser = subparsers.add_parser("process", help="Process and scrape product data for extracted URLs.")
    process_parser.add_argument("--batch-size", type=int, help="Initial batch size (will be optimized adaptively)")

    # Download stage
    download_parser = subparsers.add_parser("download", help="Download free files for eligible products.")
    download_parser.add_argument("--dry-run", action="store_true", help="List downloads without saving files.")

    # Update stage
    subparsers.add_parser("update", help="Update config metadata for product data.")

    # Performance stats stage
    stats_parser = subparsers.add_parser("stats", help="Show performance statistics.")
    stats_parser.add_argument("--filter-stage", choices=["extraction", "processing"], help="Filter by stage")
    stats_parser.add_argument("--limit", type=int, default=20, help="Number of recent records to show")

    # Backfill preview_keywords (as a separate flag, not a stage)
    parser.add_argument(
        "--backfill-preview",
        action="store_true",
        help="Backfill missing preview_keywords for product data (runs independently of stage)."
    )

    args = parser.parse_args()
    await setup_db()

    # Run backfill_preview_keywords exclusively if requested
    if args.backfill_preview:
        await backfill_preview_keywords()
        return

    if args.stage == "test":
        await extraction_test()
    elif args.stage == "extract":
        batch_size = getattr(args, "batch_size", None)
        await extraction_stage(batch_size)
    elif args.stage == "process":
        batch_size = getattr(args, "batch_size", None)
        await processing_stage(batch_size)
    elif args.stage == "download":
        await processing_free_download_stage(dry_run=getattr(args, "dry_run", False))
    elif args.stage == "update":
        await update_config_metadata()
        await check_config_metadata()
    elif args.stage == "stats":
        await show_performance_stats(
            stage=getattr(args, "filter_stage", None),
            limit=getattr(args, "limit", 20)
        )

if __name__ == "__main__":
    asyncio.run(main())