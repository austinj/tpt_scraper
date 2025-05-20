import asyncio
import aiohttp_client_cache
import async_timeout
import random
import json
import logging
import aiosqlite
import math
import os
import argparse
from pathlib import Path
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from tqdm.asyncio import tqdm_asyncio

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
            "folder_structures": ["teacher-tools/classroom-management", "teacher-tools/subject-math"],
            "price_options": ["", "on-sale"],
            "sorting_methods": ["Relevance", "Rating", "Price-Low-to-High", "Newest"],
            "total_pages": 42,
            "concurrent_requests": 10,
            "max_retries": 3,
            "sleep_between_batches": [2, 5],
            "download_sleep": [5, 10],
            "retry_backoff_start": 1,
            "retry_backoff_factor": 2
        }

config = load_config()
FOLDER_STRUCTURES = config.get("folder_structures", [])
PRICE_OPTIONS = config.get("price_options", [])
SORTING_METHODS = config.get("sorting_methods", [])
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
        await db.execute("PRAGMA journal_mode=WAL;")
        await db.execute("PRAGMA synchronous=NORMAL;")

        await db.execute("""
            CREATE TABLE IF NOT EXISTS extracted_urls (
                url TEXT PRIMARY KEY,
                folder TEXT,
                price_option TEXT,
                sort_order TEXT,
                page INTEGER
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS extracted_pages (
                folder TEXT,
                price_option TEXT,
                sort_order TEXT,
                page INTEGER,
                PRIMARY KEY(folder, price_option, sort_order, page)
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
                   MIN(folder) AS folder,
                   MIN(price_option) AS price_option,
                   MIN(sort_order) AS sort_order,
                   MIN(page) AS page
            FROM extracted_urls
            GROUP BY url
        """)
        # Updated product_data table with configuration metadata.
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
                folder TEXT,
                price_option TEXT,
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
        await db.commit()

async def insert_product_data(data):
    """
    Insert product data along with configuration parameters.
    Data should be a tuple in the form:
    (title, short_description, long_description, rating_value, number_of_ratings,
     product_price, preview_keywords, url, folder, price_option, sort_order, page, config_metadata)
    """
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute("""
            INSERT OR IGNORE INTO product_data 
            (title, short_description, long_description, rating_value, number_of_ratings, product_price, preview_keywords, url, folder, price_option, sort_order, page, config_metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, data)
        await db.commit()

async def get_unique_extracted_urls():
    async with aiosqlite.connect(DB_FILE) as db:
        async with db.execute(
            "SELECT url, folder, price_option, sort_order, page FROM unique_extracted_urls"
        ) as cursor:
            return await cursor.fetchall()

###########################
# URL Building and Fetch  #
###########################

def build_page_url(folder_structure, price_option, sort_order, page):
    """
    Builds a complete URL based on the folder structure, price option, sort order, and page number.
    For "Relevance", no order parameter is appended.
    """
    base_url = f"https://www.teacherspayteachers.com/browse/{folder_structure}"
    if price_option:
        base_url = f"{base_url}/{price_option}"
    if sort_order == "Relevance":
        return base_url if page == 1 else f"{base_url}?page={page}"
    else:
        return f"{base_url}?order={sort_order}&page={page}"

async def fetch(session, url):
    """Fetches a URL using aiohttp with retry logic and exponential backoff."""
    retries = 0
    backoff = 1
    while retries < MAX_RETRIES:
        try:
            async with async_timeout.timeout(30):
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.text()
                    else:
                        logging.warning("Non-200 status %s for URL: %s", response.status, url)
                        raise Exception(f"Status {response.status}")
        except Exception as e:
            retries += 1
            logging.warning("Error fetching %s: %s. Retry %s/%s in %s sec", url, e, retries, MAX_RETRIES, backoff)
            await asyncio.sleep(backoff)
            backoff *= 2
    logging.error("Failed to fetch %s after %s retries", url, MAX_RETRIES)
    return None

###########################
# Extraction Stage        #
###########################

async def extract_urls_from_page(session, page_url):
    """Extracts product URLs from a single page using BeautifulSoup (lxml parser)."""
    html = await fetch(session, page_url)
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

async def extract_page_wrapper(session, page_url, semaphore, folder, price_option, sort_order, page):
    async with semaphore:
        logging.info("Fetching page: %s", page_url)
        urls = await extract_urls_from_page(session, page_url)
        if urls:
            async with aiosqlite.connect(DB_FILE) as db:
                # Immediately mark the page as processed and insert URLs
                await db.execute(
                    "INSERT OR IGNORE INTO extracted_pages VALUES (?, ?, ?, ?)",
                    (folder, price_option, sort_order, page),
                )
                for url in urls:
                    await db.execute(
                        "INSERT OR IGNORE INTO extracted_urls (url, folder, price_option, sort_order, page) VALUES (?, ?, ?, ?, ?)",
                        (url, folder, price_option, sort_order, page)
                    )
                await db.commit()
        return urls

async def extraction_stage(batch_size=50):
    logging.info("Using SQLite database at %s", os.path.abspath(DB_FILE))

    # 1️⃣ Generate every combination
    combos = [
        (folder, price, sort, page)
        for folder in FOLDER_STRUCTURES
        for price in PRICE_OPTIONS
        for sort in SORTING_METHODS
        for page in range(1, TOTAL_PAGES + 1)
    ]

    # 2️⃣ Load already‑done combos from the DB
    async with aiosqlite.connect(DB_FILE) as db:
        async with db.execute("""
            SELECT folder, price_option, sort_order, page 
            FROM extracted_pages
        """) as cursor:
            done_combos = set(await cursor.fetchall())

    logging.info("Already done combos: %s/%s", len(done_combos), len(combos))

    # 3️⃣ Filter out completed combos
    remaining = [c for c in combos if c not in done_combos]
    total_batches = math.ceil(len(remaining) / batch_size)
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    async with aiohttp_client_cache.CachedSession(cache_name="aiohttp_cache", expire_after=3600) as session:
        for batch_idx in range(total_batches):
            start, end = batch_idx * batch_size, (batch_idx + 1) * batch_size
            batch = remaining[start:end]
            tasks = [
                extract_page_wrapper(session,
                                     build_page_url(folder, price, sort, page),
                                     semaphore,
                                     folder, price, sort, page)
                for folder, price, sort, page in batch
            ]

            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                logging.error("Error processing batch %s: %s", batch_idx, e)

            logging.info("Batch %s/%s complete.", batch_idx + 1, total_batches)
            await asyncio.sleep(random.uniform(2, 5))

    logging.info("Resumed extraction finished; all new pages saved.")

###########################
# Extraction Test Stage   #
###########################

async def extraction_test(test_limit=5):
    """
    Test extraction stage that fetches a single page from a single combination,
    stores up to test_limit URLs with their parameters in the SQLite database,
    and prints them.
    """
    extracted_url_details = {}  # dict mapping url -> (folder, price_option, sort_order, page)
    async with aiohttp_client_cache.CachedSession(cache_name="aiohttp_cache", expire_after=3600) as session:
        folder = FOLDER_STRUCTURES[0] if FOLDER_STRUCTURES else "teacher-tools/classroom-management"
        price = PRICE_OPTIONS[0] if PRICE_OPTIONS else ""
        sort_order = SORTING_METHODS[0] if SORTING_METHODS else "Relevance"
        page = 1
        page_url = build_page_url(folder, price, sort_order, page)
        logging.info("Test fetching page: %s", page_url)
        urls = await extract_urls_from_page(session, page_url)
        for url in urls:
            if len(extracted_url_details) >= test_limit:
                break
            extracted_url_details[url] = (folder, price, sort_order, page)
    logging.info("Test extracted %s URLs", len(extracted_url_details))
    async with aiosqlite.connect(DB_FILE) as db:
        for url, params in extracted_url_details.items():
            folder, price_option, sort_order, page = params
            await db.execute("""
                INSERT OR IGNORE INTO extracted_urls (url, folder, price_option, sort_order, page)
                VALUES (?, ?, ?, ?, ?)
            """, (url, folder, price_option, sort_order, page))
        await db.commit()
    logging.info("Test extracted URLs saved to SQLite database (%s).", DB_FILE)
    for url, params in extracted_url_details.items():
        print(f"{url} -> {params}")

###########################
# Processing Stage        #
###########################

async def scrape_product_data(session, url):
    """Scrapes product data from a product URL using BeautifulSoup (lxml parser)."""
    html = await fetch(session, url)
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
    meta_price = soup.find("meta", {"name": "price"})
    product_price = meta_price["content"] if meta_price and meta_price.has_attr("content") else None

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

async def processing_stage(batch_size=50):
    url_records = await get_unique_extracted_urls()  # Each record is (url, folder, price_option, sort_order, page)
    total_records = len(url_records)
    logging.info("Total unique URLs to process: %s", total_records)

    async with aiohttp_client_cache.CachedSession(cache_name="aiohttp_cache", expire_after=3600) as session:
        for i in range(0, total_records, batch_size):
            batch = url_records[i:i + batch_size]
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
                    tasks.append(scrape_product_data(session, url))
                    indices_to_process.append(idx)

            pre_skipped = len(batch) - len(tasks)  # Count of URLs pre-skipped from the batch.
            results = await asyncio.gather(*tasks, return_exceptions=True) if tasks else []
            new_count = 0
            skipped_count = pre_skipped  # Start with pre-skipped URLs as skipped.

            # Insert newly scraped data into the DB.
            async with aiosqlite.connect(DB_FILE) as db:
                for result, idx in zip(results, indices_to_process):
                    if result and not isinstance(result, Exception):
                        # Retrieve associated parameters from the batch record.
                        url, folder, price_option, sort_order, page = batch[idx]
                        # Build a configuration metadata JSON string from the extraction parameters.
                        config_metadata = json.dumps({
                            "folder": folder,
                            "price_option": price_option,
                            "sort_order": sort_order,
                            "page": page
                        })
                        full_data = (*result, folder, price_option, sort_order, page, config_metadata)
                        cursor = await db.execute(
                            """
                            INSERT OR IGNORE INTO product_data 
                            (title, short_description, long_description, rating_value, number_of_ratings, product_price, preview_keywords, url, folder, price_option, sort_order, page, config_metadata)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, full_data
                        )
                        if cursor.rowcount > 0:
                            new_count += 1
                        else:
                            skipped_count += 1
                    else:
                        skipped_count += 1
                await db.commit()

            batch_complete = i + len(batch)
            percent_complete = (batch_complete / total_records) * 100
            logging.info(
                "Processed batch %s (%s to %s): %s new, %s skipped. %.2f%% complete.",
                (i // batch_size) + 1,
                i + 1,
                i + len(batch),
                new_count,
                skipped_count,
                percent_complete
            )

            # Only sleep if there was any new data inserted.
            if new_count > 0:
                await asyncio.sleep(random.uniform(5, 10))
    logging.info("Finished processing unique URLs.")

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
    based on the existing folder, price_option, sort_order, and page values.
    """
    # First, ensure the column exists.
    await add_config_metadata_column_if_needed()
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute("""
            UPDATE product_data
            SET config_metadata = json_object(
                'folder', folder,
                'price_option', price_option,
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
        
        async with db.execute("SELECT id, folder, price_option, sort_order, page, config_metadata FROM product_data LIMIT 5") as cursor:
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

    total = len(rows)
    sem = asyncio.Semaphore(concurrency)
    async with aiohttp_client_cache.CachedSession(cache_name="aiohttp_cache", expire_after=3600) as session:
        async with aiosqlite.connect(DB_FILE) as db:  # Reuse connection
            for i in range(0, total, batch_size):
                batch = rows[i:i+batch_size]
                async def scrape_with_sem(url):
                    async with sem:
                        return await scrape_product_data(session, url)
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

###########################
# Main Entry Point        #
###########################

async def main():
    parser = argparse.ArgumentParser(
        description="TPT Scraper Tool: Extract, process, and download TPT product data."
    )
    subparsers = parser.add_subparsers(dest="stage", required=True, help="Stage to run")

    # Test stage
    subparsers.add_parser("test", help="Test extraction on a single page.")

    # Extract stage
    subparsers.add_parser("extract", help="Extract product URLs for all config combinations.")

    # Process stage
    subparsers.add_parser("process", help="Process and scrape product data for extracted URLs.")

    # Download stage
    download_parser = subparsers.add_parser("download", help="Download free files for eligible products.")
    download_parser.add_argument("--dry-run", action="store_true", help="List downloads without saving files.")

    # Update stage
    subparsers.add_parser("update", help="Update config metadata for product data.")

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
        await extraction_stage()
    elif args.stage == "process":
        await processing_stage()
    elif args.stage == "download":
        await processing_free_download_stage(dry_run=getattr(args, "dry_run", False))
    elif args.stage == "update":
        await update_config_metadata()
        await check_config_metadata()

if __name__ == "__main__":
    asyncio.run(main())
