import asyncio
import aiohttp_client_cache
import async_timeout
import random
import json
import logging
import aiosqlite
from bs4 import BeautifulSoup

# Configuration file name and SQLite database file
CONFIG_FILE = "config.json"
DB_FILE = "scrape_cache.db"

def load_config(config_file=CONFIG_FILE):
    """Loads configuration from an external JSON file."""
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config
    except Exception as e:
        logging.error("Error loading config: %s", e)
        # Provide defaults if file not found or on error
        return {
            "folder_structures": [
                "teacher-tools/classroom-management",
                "teacher-tools/subject-math"
            ],
            "price_options": ["", "on-sale"],
            "sorting_methods": ["Relevance", "Rating", "Price-Low-to-High", "Newest"],
            "total_pages": 42,
            "concurrent_requests": 10,
            "max_retries": 3
        }

# Load external configuration
config = load_config()
FOLDER_STRUCTURES = config.get("folder_structures", [])
PRICE_OPTIONS = config.get("price_options", [])
SORTING_METHODS = config.get("sorting_methods", [])
TOTAL_PAGES = config.get("total_pages", 42)
CONCURRENT_REQUESTS = config.get("concurrent_requests", 10)
MAX_RETRIES = config.get("max_retries", 3)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

###########################
# SQLite Database Helpers #
###########################

async def setup_db():
    """Creates the SQLite database and tables if they don't exist."""
    async with aiosqlite.connect(DB_FILE) as db:
        # Create extracted_urls table with configuration parameters
        await db.execute("""
            CREATE TABLE IF NOT EXISTS extracted_urls (
                url TEXT PRIMARY KEY,
                folder TEXT,
                price_option TEXT,
                sort_order TEXT,
                page INTEGER
            )
        """)
        # Create product_data table with scraped product fields and configuration parameters.
        await db.execute("""
            CREATE TABLE IF NOT EXISTS product_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                short_description TEXT,
                long_description TEXT,
                rating_value TEXT,
                number_of_ratings TEXT,
                product_price TEXT,
                url TEXT UNIQUE,
                folder TEXT,
                price_option TEXT,
                sort_order TEXT,
                page INTEGER
            )
        """)
        await db.commit()

async def insert_extracted_urls(url_details):
    """
    Insert extracted URLs along with their configuration parameters.
    `url_details` should be a dict mapping url -> (folder, price_option, sort_order, page)
    """
    async with aiosqlite.connect(DB_FILE) as db:
        for url, params in url_details.items():
            folder, price_option, sort_order, page = params
            await db.execute("""
                INSERT OR IGNORE INTO extracted_urls (url, folder, price_option, sort_order, page)
                VALUES (?, ?, ?, ?, ?)
            """, (url, folder, price_option, sort_order, page))
        await db.commit()

async def get_extracted_urls():
    """
    Retrieve all URLs with their configuration parameters.
    Returns a list of tuples: (url, folder, price_option, sort_order, page)
    """
    async with aiosqlite.connect(DB_FILE) as db:
        async with db.execute("SELECT url, folder, price_option, sort_order, page FROM extracted_urls") as cursor:
            rows = await cursor.fetchall()
    return rows

async def insert_product_data(data):
    """
    Insert product data along with configuration parameters.
    Data should be a tuple in the form:
    (title, short_description, long_description, rating_value, number_of_ratings, product_price, url, folder, price_option, sort_order, page)
    """
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute("""
            INSERT OR IGNORE INTO product_data 
            (title, short_description, long_description, rating_value, number_of_ratings, product_price, url, folder, price_option, sort_order, page)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, data)
        await db.commit()

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

async def extract_page_wrapper(session, page_url, semaphore, results, folder, price_option, sort_order, page):
    """Wrapper to control concurrency and update the results dict with configuration parameters."""
    async with semaphore:
        logging.info("Fetching page: %s", page_url)
        urls = await extract_urls_from_page(session, page_url)
        if urls:
            for url in urls:
                # Save the config parameters alongside the URL if not already present
                if url not in results:
                    results[url] = (folder, price_option, sort_order, page)
        return urls

async def extraction_stage():
    """
    Extraction stage:
    Asynchronously fetch pages for every combination, deduplicate URLs (with parameters),
    and store them in SQLite.
    """
    extracted_url_details = {}  # dict mapping url -> (folder, price_option, sort_order, page)
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    tasks = []
    async with aiohttp_client_cache.CachedSession(cache_name="aiohttp_cache", expire_after=3600) as session:
        for folder in FOLDER_STRUCTURES:
            for price in PRICE_OPTIONS:
                for sort_order in SORTING_METHODS:
                    for page in range(1, TOTAL_PAGES + 1):
                        page_url = build_page_url(folder, price, sort_order, page)
                        tasks.append(extract_page_wrapper(session, page_url, semaphore, extracted_url_details, folder, price, sort_order, page))
        await asyncio.gather(*tasks, return_exceptions=True)
    logging.info("Total unique URLs extracted: %s", len(extracted_url_details))
    await insert_extracted_urls(extracted_url_details)
    logging.info("Extracted URLs saved to SQLite database (%s).", DB_FILE)

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
    await insert_extracted_urls(extracted_url_details)
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
    return title, short_description, long_description, rating_value, number_of_ratings, product_price, url

async def processing_stage(batch_size=50):
    """
    Processing stage:
    Reads URLs with their configuration parameters from SQLite, scrapes product data concurrently,
    and stores the results (combined with the config parameters) in the database.
    """
    # Retrieve rows as tuples: (url, folder, price_option, sort_order, page)
    url_records = await get_extracted_urls()
    total_records = len(url_records)
    logging.info("Total URLs to process: %s", total_records)
    async with aiohttp_client_cache.CachedSession(cache_name="aiohttp_cache", expire_after=3600) as session:
        # Process in batches
        for i in range(0, total_records, batch_size):
            batch = url_records[i:i+batch_size]
            tasks = []
            # Each record contains: (url, folder, price_option, sort_order, page)
            for record in batch:
                url, folder, price_option, sort_order, page = record
                tasks.append(scrape_product_data(session, url))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for j, result in enumerate(results):
                if result and not isinstance(result, Exception):
                    # Retrieve the config parameters for this URL from the batch
                    url, folder, price_option, sort_order, page = batch[j]
                    # Combine scraped data with config parameters.
                    # result is a tuple: (title, short_description, long_description, rating_value, number_of_ratings, product_price, url)
                    full_data = result + (folder, price_option, sort_order, page)
                    await insert_product_data(full_data)
            logging.info("Processed batch %s (%s to %s)", (i // batch_size) + 1, i+1, min(i+batch_size, total_records))
            await asyncio.sleep(random.uniform(5.0, 10.0))  # Throttle between batches
    logging.info("Product data stored in SQLite database (%s).", DB_FILE)

###########################
# Main Entry Point        #
###########################

async def main():
    # Note: Using the built-in input() in an async function blocks the event loop.
    # For advanced applications, consider an asynchronous alternative.
    await setup_db()  # Ensure our database and tables are created
    print("Choose an option:")
    print("0. Test Extraction Stage (Extract 5 URLs)")
    print("1. Full Extraction Stage: Extract and store product URLs (SQLite)")
    print("2. Processing Stage: Scrape product data from stored URLs and save in SQLite")
    choice = input("Enter 0, 1 or 2: ").strip()
    if choice == "0":
        await extraction_test()
    elif choice == "1":
        await extraction_stage()
    elif choice == "2":
        await processing_stage()
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    asyncio.run(main())
