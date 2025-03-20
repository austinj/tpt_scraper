import asyncio
import aiohttp
import async_timeout
import csv
import os
import time
import random
import json
import logging
from bs4 import BeautifulSoup

CONFIG_FILE = "config.json"

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

async def extraction_stage(output_csv="output_url/extracted_urls.csv"):
    """
    Extraction stage:
    Asynchronously fetch pages for every combination, deduplicate URLs, and write them incrementally to CSV.
    """
    unique_urls = set()
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession() as session:
        tasks = []
        # Schedule tasks for each combination and page
        for folder in FOLDER_STRUCTURES:
            for price in PRICE_OPTIONS:
                for sort_order in SORTING_METHODS:
                    for page in range(1, TOTAL_PAGES + 1):
                        page_url = build_page_url(folder, price, sort_order, page)
                        tasks.append(extract_page_wrapper(session, page_url, unique_urls, semaphore))
        await asyncio.gather(*tasks, return_exceptions=True)
    logging.info("Total unique URLs extracted: %s", len(unique_urls))
    # Write deduplicated URLs to CSV (ensure output folder exists)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Product URL"])
        for url in unique_urls:
            writer.writerow([url])
    logging.info("Extracted URLs saved to %s", output_csv)

async def extract_page_wrapper(session, page_url, unique_urls, semaphore):
    """Wrapper to control concurrency and update the unique URL set."""
    async with semaphore:
        logging.info("Fetching page: %s", page_url)
        urls = await extract_urls_from_page(session, page_url)
        if urls:
            for url in urls:
                unique_urls.add(url)
        return urls

async def extraction_test(test_limit=5):
    """
    Test extraction stage that fetches a single page from a single combination and prints up to test_limit URLs.
    """
    unique_urls = set()
    async with aiohttp.ClientSession() as session:
        # Use the first combination and the first page as a test.
        folder = FOLDER_STRUCTURES[0] if FOLDER_STRUCTURES else "teacher-tools/classroom-management"
        price = PRICE_OPTIONS[0] if PRICE_OPTIONS else ""
        sort_order = SORTING_METHODS[0] if SORTING_METHODS else "Relevance"
        page_url = build_page_url(folder, price, sort_order, 1)
        logging.info("Test fetching page: %s", page_url)
        urls = await extract_urls_from_page(session, page_url)
        for url in urls:
            unique_urls.add(url)
            if len(unique_urls) >= test_limit:
                break
    logging.info("Test extracted %s URLs", len(unique_urls))
    for url in unique_urls:
        print(url)

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
    price = meta_price["content"] if meta_price and meta_price.has_attr("content") else None
    return title, short_description, long_description, rating_value, number_of_ratings, price, url

async def processing_stage(input_csv="output_url/extracted_urls.csv", output_csv="output_data/product_data.csv", batch_size=50):
    """
    Processing stage:
    Reads URLs from CSV in batches, scrapes product data concurrently, and writes results incrementally to CSV.
    """
    urls = []
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if row and row[0].startswith("http"):
                urls.append(row[0])
    total_urls = len(urls)
    logging.info("Total URLs to process: %s", total_urls)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    async with aiohttp.ClientSession() as session:
        with open(output_csv, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.writer(f_out, quoting=csv.QUOTE_ALL)
            writer.writerow(["Title", "Short Description", "Long Description", "Overall Rating", "Number of Ratings", "Price", "URL"])
            # Process in batches
            for i in range(0, total_urls, batch_size):
                batch = urls[i:i+batch_size]
                tasks = [scrape_product_data(session, url) for url in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if result and not isinstance(result, Exception):
                        writer.writerow(result)
                logging.info("Processed batch %s (%s to %s)", (i // batch_size) + 1, i+1, min(i+batch_size, total_urls))
                await asyncio.sleep(random.uniform(5.0, 10.0))  # Throttle between batches

async def main():
    print("Choose an option:")
    print("0. Test Extraction Stage (Extract 5 URLs)")
    print("1. Full Extraction Stage: Extract and store product URLs")
    print("2. Processing Stage: Scrape product data from stored URLs")
    choice = input("Enter 0, 1 or 2: ").strip()
    if choice == "0":
        await extraction_test()
    elif choice == "1":
        await extraction_stage()
    elif choice == "2":
        # Allow user to specify the CSV file to process; default is in output_url folder
        input_csv = input("Enter CSV file of URLs to process (default: output_url/extracted_urls.csv): ").strip()
        if not input_csv:
            input_csv = "output_url/extracted_urls.csv"
        output_csv = input("Enter output CSV file for product data (default: output_data/product_data.csv): ").strip()
        if not output_csv:
            output_csv = "output_data/product_data.csv"
        await processing_stage(input_csv, output_csv)
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    asyncio.run(main())
