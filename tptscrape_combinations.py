import csv
import os
import time
import random
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException

CONFIG_FILE = "config.json"

def load_config(config_file=CONFIG_FILE):
    """Loads configuration from an external JSON file."""
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Configuration file '{config_file}' not found. Using default configuration.")
        return {
            "folder_structures": [
                "teacher-tools/classroom-management",
                "teacher-tools/subject-math"
            ],
            "price_options": ["", "on-sale"],
            "sorting_methods": ["Relevance", "Rating", "Price-Low-to-High", "Newest"]
        }

# Load external configuration
config = load_config()
FOLDER_STRUCTURES = config.get("folder_structures", [])
PRICE_OPTIONS = config.get("price_options", [])
SORTING_METHODS = config.get("sorting_methods", [])

def safe_quit(driver):
    """Safely quits the driver, ignoring benign OSError (WinError 6)."""
    try:
        driver.quit()
    except OSError as e:
        if getattr(e, 'errno', None) == 6:
            pass
        else:
            raise

def create_driver():
    """Creates a headless Selenium Chrome WebDriver instance."""
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    driver = webdriver.Chrome(options=options)
    return driver

def extract_urls_from_page(driver, page_url):
    """Extracts product URLs from a single page."""
    driver.get(page_url)
    time.sleep(random.uniform(1.0, 3.0))
    wait = WebDriverWait(driver, 30)
    
    try:
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'a.ProductRowCard-module__cardTitleLink--YPqiC')))
    except TimeoutException:
        print("Error: Timeout while waiting for product elements to load.")
        return []

    product_elements = driver.find_elements(By.CSS_SELECTOR, 'a.ProductRowCard-module__cardTitleLink--YPqiC')
    product_urls = []
    for product in product_elements:
        url = product.get_attribute('href')
        if url:
            full_url = "https://www.teacherspayteachers.com" + url if not url.startswith("http") else url
            product_urls.append(full_url)
            print(f"Extracted URL: {full_url}")
    
    return product_urls

def extract_urls_from_all_pages(total_pages=42):
    """Extracts product URLs across all pages for a single hardcoded combination."""
    base_url = "https://www.teacherspayteachers.com/browse/teacher-tools/classroom-management"
    all_urls = []
    print("Initializing WebDriver...")
    driver = create_driver()
    print("WebDriver initialized.")

    try:
        for page in range(1, total_pages + 1):
            page_url = f"{base_url}?order=Rating-Count&page={page}"
            print(f"Processing page {page}: {page_url}")
            urls_from_page = extract_urls_from_page(driver, page_url)
            if not urls_from_page:
                print(f"No products found on page {page}. Stopping pagination.")
                break
            all_urls.extend(urls_from_page)
            time.sleep(random.uniform(2.0, 5.0))
    finally:
        safe_quit(driver)

    return all_urls

def build_page_url(folder_structure, price_option, sort_order, page):
    """
    Builds a complete URL based on the folder structure, price option, sort order, and page number.
    If a price option is provided (e.g., "on-sale"), it is appended to the URL path.
    
    When sort_order is "Relevance", the URL omits the ?order= argument:
      - For page 1, it returns just the base URL.
      - For subsequent pages, it appends ?page={page}.
    Otherwise, it appends ?order={sort_order}&page={page}.
    """
    base_url = f"https://www.teacherspayteachers.com/browse/{folder_structure}"
    if price_option:
        base_url = f"{base_url}/{price_option}"
    
    if sort_order == "Relevance":
        # For the default sorting method (Relevance), only add page if page > 1.
        return base_url if page == 1 else f"{base_url}?page={page}"
    else:
        return f"{base_url}?order={sort_order}&page={page}"

def extract_urls_for_combinations(total_pages=42, folder_structures=FOLDER_STRUCTURES, price_options=PRICE_OPTIONS, sorting_methods=SORTING_METHODS):
    """
    Extracts product URLs over multiple combinations of folder structures, price options, and sorting methods.
    """
    all_urls = []
    print("Initializing WebDriver for combinations...")
    driver = create_driver()
    print("WebDriver initialized for combinations.")
    
    try:
        for folder in folder_structures:
            for price_option in price_options:
                for sort_order in sorting_methods:
                    print(f"\nStarting extraction for Folder: '{folder}', Price: '{price_option or 'regular'}', with Sort Order: '{sort_order}'")
                    for page in range(1, total_pages + 1):
                        page_url = build_page_url(folder, price_option, sort_order, page)
                        print(f"Processing page {page}: {page_url}")
                        urls_from_page = extract_urls_from_page(driver, page_url)
                        if not urls_from_page:
                            print(f"No products found on page {page} for folder '{folder}', price '{price_option or 'regular'}' and sort '{sort_order}'.")
                            continue
                        all_urls.extend(urls_from_page)
                        time.sleep(random.uniform(2.0, 5.0))
    finally:
        safe_quit(driver)
    
    return all_urls

def save_urls_to_csv(urls, csv_file):
    """Saves a list of URLs to a CSV file."""
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Product URL'])
        for url in urls:
            writer.writerow([url])
    print(f"URLs saved to {csv_file}")

def scrape_product_data(driver, url):
    """Scrapes detailed product data from a given product URL."""
    driver.get(url)
    time.sleep(random.uniform(1.0, 3.0))
    driver.implicitly_wait(10)

    title = None
    short_description = None
    long_description = None
    rating_value, number_of_ratings = None, None
    price = None

    try:
        title = driver.title
        short_description = driver.find_element(By.XPATH, "//meta[@name='description']").get_attribute('content')
        long_description = driver.find_element(By.CLASS_NAME, 'ProductDescriptionLayout__htmlDisplay').text

        rating_section = driver.find_element(By.CSS_SELECTOR, 'span.StarRating-module__srOnly--FAzEA')
        rating_text = rating_section.get_attribute('innerText')
        parts = rating_text.split(' ')
        if len(parts) >= 3:
            rating_value = parts[1]
            number_of_ratings = parts[-2]

        price = driver.find_element(By.XPATH, "//meta[@name='price']").get_attribute('content')
    except Exception:
        pass

    return title, short_description, long_description, rating_value, number_of_ratings, price, url

def process_urls_in_batches(input_csv, output_csv, batch_size=50):
    """Processes URLs from an input CSV in batches and scrapes product data."""
    print(f"Reading input CSV: {input_csv}")
    urls = []
    with open(input_csv, mode='r', encoding='utf-8') as url_file:
        reader = csv.reader(url_file)
        header = next(reader)  # Skip header row
        print(f"Skipping header row: {header}")
        for row in reader:
            if row and row[0].startswith("http"):
                urls.append(row[0])
            else:
                print(f"Skipping invalid row: {row}")

    total_urls = len(urls)
    print(f"Total URLs to process: {total_urls}")

    if total_urls == 0:
        print("No valid URLs found. Exiting.")
        return

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL)
        writer.writerow(['Title', 'Short Description', 'Long Description', 'Overall Rating', 'Number of Ratings',
                         'Price', 'URL'])

    failed_urls = []

    for batch_start in range(0, total_urls, batch_size):
        batch = urls[batch_start: batch_start + batch_size]
        print(f"Processing batch {(batch_start // batch_size) + 1} ({batch_start + 1} to {min(batch_start + batch_size, total_urls)})")
        driver = create_driver()
        batch_results = []

        for index, url in enumerate(batch, start=batch_start + 1):
            print(f"🔍 Scraping {index}/{total_urls}: {url}")
            try:
                product_data = scrape_product_data(driver, url)
                print(f"✅ Successfully scraped: {url}")
                batch_results.append(product_data)
            except Exception as e:
                print(f"❌ Error processing {url}: {e}")
                failed_urls.append(url)
            time.sleep(random.uniform(2.0, 4.0))
        safe_quit(driver)

        with open(output_csv, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)
            writer.writerows(batch_results)
        print(f"📌 Finished batch {(batch_start // batch_size) + 1}")
        time.sleep(random.uniform(5.0, 10.0))

    if failed_urls:
        failed_urls_file = "failed_urls.txt"
        with open(failed_urls_file, "w") as f:
            f.write("\n".join(failed_urls) + "\n")
        print(f"❗ {len(failed_urls)} URLs failed. Saved in '{failed_urls_file}'.")

def retry_failed_urls(output_csv, batch_size=50):
    """Retries processing URLs stored in 'failed_urls.txt'."""
    failed_urls_file = "failed_urls.txt"
    if not os.path.exists(failed_urls_file):
        print("✅ No failed URLs to retry.")
        return

    with open(failed_urls_file, "r") as f:
        urls = [line.strip() for line in f.readlines() if line.strip()]

    if not urls:
        print("✅ No failed URLs to retry.")
        return

    print(f"🔄 Retrying {len(urls)} failed URLs...")
    process_urls_in_batches(failed_urls_file, output_csv, batch_size)

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Extract product URLs (Single combination)")
    print("2. Process an existing CSV file of URLs")
    print("3. Retry failed URLs")
    print("4. Extract URLs over multiple combinations")
    choice = input("Enter 1, 2, 3, or 4: ").strip()

    if choice == "1":
        output_csv = "URL_List/classroom_management_urls.csv"
        urls = extract_urls_from_all_pages()
        save_urls_to_csv(urls, output_csv)

    elif choice == "2":
        process_urls_in_batches("URL_List/classroom_management_urls.csv", "Info/classroom_management_data.csv")

    elif choice == "3":
        retry_failed_urls("Info/retried_failed_urls.csv")

    elif choice == "4":
        output_csv = "URL_List/combined_urls.csv"
        urls = extract_urls_for_combinations(total_pages=42)
        save_urls_to_csv(urls, output_csv)
    
    else:
        print("Invalid choice. Please restart the script and enter 1, 2, 3, or 4.")
