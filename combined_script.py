import csv
import os
import time
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException

def safe_quit(driver):
    """
    Safely quits the driver, ignoring benign OSError (WinError 6).
    """
    try:
        driver.quit()
    except OSError as e:
        if getattr(e, 'errno', None) == 6:
            pass
        else:
            raise

def create_driver():
    """
    Creates a standard Selenium Chrome WebDriver instance with headless mode enabled.
    """
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    
    driver = webdriver.Chrome(options=options)
    return driver

def retry_on_stale_element(func, *args, max_retries=3, wait_time=1, **kwargs):
    """
    Attempts to execute a function and retries if a StaleElementReferenceException occurs.
    """
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except StaleElementReferenceException:
            print(f"Warning: Stale element reference on attempt {attempt + 1}. Retrying...")
            time.sleep(wait_time)
    print("Error: Could not complete operation due to repeated stale element reference exceptions.")
    return None

def extract_urls_from_page(driver, page_url):
    """
    Extracts product URLs from a single page.
    """
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
    if product_elements:
        for product in product_elements:
            url = product.get_attribute('href')
            if url:
                full_url = "https://www.teacherspayteachers.com" + url if not url.startswith("http") else url
                product_urls.append(full_url)
                print(f"Extracted URL: {full_url}")
    return product_urls

def extract_urls_from_all_pages(keyword, total_pages):
    """
    Extracts product URLs across the requested number of pages for a given keyword.
    """
    base_url = "https://www.teacherspayteachers.com/browse/teacher-tools/classroom-management"
    all_urls = []
    driver = create_driver()

    try:
        for page in range(1, total_pages + 1):
            page_url = f"{base_url}?order=Rating-Count&page={page}&search={keyword}"
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

def save_urls_to_csv(urls, csv_file):
    """
    Saves a list of URLs to a CSV file within the 'URL_List' directory.
    """
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)  # Ensure directory exists
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Product URL'])
        for url in urls:
            writer.writerow([url])
    print(f"URLs saved to {csv_file}")

def scrape_product_data(driver, url):
    """
    Scrapes detailed product data from a given product URL.
    """
    driver.get(url)
    # Random delay
    time.sleep(random.uniform(1.0, 3.0))
    driver.implicitly_wait(10)

    # Extract grade level and category data
    grade_level = None
    categories = []

    grade_element = retry_on_stale_element(driver.find_element, By.CSS_SELECTOR,
                                             '[data-testid="RebrandedContentText"] .NotLinkedSection span')
    if grade_element:
        grade_level = grade_element.text

    category_elements = retry_on_stale_element(driver.find_elements, By.CSS_SELECTOR,
                                               'div[data-testid="LabeledSectionContent"] a.Link-module__link--GFbUH')
    if category_elements:
        categories = [category.text for category in category_elements]

    preview_keywords = f"{grade_level} " + ", ".join(categories)

    title = retry_on_stale_element(lambda: driver.title)
    short_description = retry_on_stale_element(driver.find_element, By.XPATH,
                                               "//meta[@name='description']").get_attribute('content')
    long_description = retry_on_stale_element(driver.find_element, By.CLASS_NAME,
                                               'ProductDescriptionLayout__htmlDisplay').text

    rating_value, number_of_ratings = None, None
    rating_section = retry_on_stale_element(driver.find_element, By.CSS_SELECTOR,
                                            'span.StarRating-module__srOnly--FAzEA')
    if rating_section:
        rating_text = rating_section.get_attribute('innerText')
        parts = rating_text.split(' ')
        if len(parts) >= 3:
            rating_value = parts[1]
            number_of_ratings = parts[-2]

    price = retry_on_stale_element(driver.find_element, By.XPATH, "//meta[@name='price']").get_attribute('content')

    return title, short_description, long_description, rating_value, number_of_ratings, price, preview_keywords, url

def process_urls_in_batches(input_csv, output_csv, search_keyword, batch_size=50):
    """
    Processes product URLs in batches from the input CSV and scrapes product details.
    Failed URLs are logged into 'failed_urls.txt' for retrying later.
    """
    urls = []
    with open(input_csv, mode='r', encoding='utf-8') as url_file:
        reader = csv.reader(url_file)
        next(reader)  # Skip header row.
        for row in reader:
            urls.append(row[0])

    total_urls = len(urls)
    print(f"Total URLs to process: {total_urls}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Log failed URLs for retrying
    failed_urls_file = "failed_urls.txt"
    failed_urls = []  # Store failed URLs in memory before writing

    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL)
        writer.writerow(
            ['Title', 'Short Description', 'Long Description', 'Overall Rating', 'Number of Ratings',
             'Price', 'Preview Keywords', 'URL', 'Search Keyword'])

    for batch_start in range(0, total_urls, batch_size):
        batch = urls[batch_start: batch_start + batch_size]
        print(f"Processing batch {(batch_start // batch_size) + 1} "
              f"(URLs {batch_start + 1} to {min(batch_start + batch_size, total_urls)})")

        driver = create_driver()
        batch_results = []

        for i, url in enumerate(batch, start=batch_start + 1):
            print(f"Processing entry {i}: {url}")
            try:
                product_data = scrape_product_data(driver, url)
                batch_results.append(product_data + (search_keyword,))
            except Exception as e:
                print(f"❌ Error processing entry {i} (URL: {url}): {e}")
                failed_urls.append(url)  # Store failed URL in memory

            time.sleep(random.uniform(2.0, 4.0))

        safe_quit(driver)

        with open(output_csv, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)
            for row in batch_results:
                writer.writerow(row)

        time.sleep(random.uniform(5.0, 10.0))

    # Save all failed URLs at the end to avoid multiple writes
    if failed_urls:
        with open(failed_urls_file, "w") as f:
            f.write("\n".join(failed_urls) + "\n")  # Write all failed URLs at once

    print(f"✅ All product details saved to {output_csv}")
    print(f"❗ Failed URLs (if any): {len(failed_urls)}. Saved in '{failed_urls_file}'. You can retry them later.")

def retry_failed_urls(output_csv, search_keyword, batch_size=50):
    """
    Retries processing URLs stored in 'failed_urls.txt'.
    If a URL fails again, it remains in 'failed_urls.txt' for another retry.
    """
    failed_urls_file = "failed_urls.txt"

    # Check if failed_urls.txt exists
    if not os.path.exists(failed_urls_file):
        print("✅ No failed URLs to retry.")
        return

    # Read failed URLs
    with open(failed_urls_file, "r") as f:
        urls = [line.strip() for line in f.readlines() if line.strip()]

    if not urls:
        print("✅ No failed URLs to retry.")
        return

    print(f"🔄 Retrying {len(urls)} failed URLs...")

    remaining_failed_urls = []

    for batch_start in range(0, len(urls), batch_size):
        batch = urls[batch_start: batch_start + batch_size]
        print(f"Processing retry batch {(batch_start // batch_size) + 1} "
              f"(URLs {batch_start + 1} to {min(batch_start + batch_size, len(urls))})")

        driver = create_driver()
        batch_results = []

        for i, url in enumerate(batch, start=batch_start + 1):
            print(f"Retrying entry {i}: {url}")
            try:
                product_data = scrape_product_data(driver, url)
                batch_results.append(product_data + (search_keyword,))
            except Exception as e:
                print(f"❌ Failed again: {url} - {e}")
                remaining_failed_urls.append(url)  # Keep URL for another retry

            time.sleep(random.uniform(2.0, 4.0))

        safe_quit(driver)

        # Append successful results to CSV
        with open(output_csv, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)
            for row in batch_results:
                writer.writerow(row)

        time.sleep(random.uniform(5.0, 10.0))

    # Overwrite failed_urls.txt with only those that failed again
    if remaining_failed_urls:
        with open(failed_urls_file, "w") as f:
            f.write("\n".join(remaining_failed_urls) + "\n")
        print(f"❗ {len(remaining_failed_urls)} URLs still failed. Check '{failed_urls_file}' to retry them again.")
    else:
        os.remove(failed_urls_file)  # Delete if all succeeded
        print("✅ All failed URLs have been successfully processed!")

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Search for products and save URLs")
    print("2. Process an existing CSV file of URLs")
    print("3. Retry failed URLs")
    choice = input("Enter 1, 2, or 3: ").strip()

    if choice == "1":
        keywords_input = input("Enter search keywords separated by commas: ")
        keywords = [keyword.strip().replace(" ", "%20") for keyword in keywords_input.split(",")]

        scrape_option = input("Scrape (A) all results (42 pages) or (B) a limited number of pages? (A/B): ").strip().upper()
        total_pages = 42 if scrape_option == "A" else int(input("Enter the number of pages to scrape (1-42): ").strip())

        for keyword in keywords:
            url_csv_file = os.path.join('URL_List', f'{keyword.replace("%20", "_")}.csv')
            print(f"Extracting URLs for keyword: '{keyword}'")
            product_urls = extract_urls_from_all_pages(keyword, total_pages)
            save_urls_to_csv(product_urls, url_csv_file)

    elif choice == "2":
        print("\nAvailable CSV files in URL_List:")
        csv_files = [f for f in os.listdir("URL_List") if f.endswith(".csv")]
        for i, file in enumerate(csv_files, 1):
            print(f"{i}. {file}")

        file_choice = int(input("Enter the number of the CSV file to process: ")) - 1
        input_csv = os.path.join("URL_List", csv_files[file_choice])
        output_csv = os.path.join("Info", csv_files[file_choice])

        batch_process = input(f"Process '{csv_files[file_choice]}' in batches? (Y/N): ").strip().upper()
        batch_size = 50 if batch_process == "Y" else None
        process_urls_in_batches(input_csv, output_csv, csv_files[file_choice], batch_size=batch_size)

    elif choice == "3":
        output_csv = "retried_failed_urls.csv"
        retry_failed_urls(output_csv, "Retrying previously failed URLs", batch_size=50)

    else:
        print("Invalid choice. Please restart the script and enter 1, 2, or 3.")