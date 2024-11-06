import csv
import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException

# Retry function to handle stale element references globally
def retry_on_stale_element(func, *args, max_retries=3, wait_time=1, **kwargs):
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except StaleElementReferenceException:
            print(f"Warning: Stale element reference on attempt {attempt + 1}. Retrying...")
            time.sleep(wait_time)
    print("Error: Could not complete operation due to repeated stale element reference exceptions.")
    return None

# Function to extract URLs from a single page with retries
def extract_urls_from_page(driver, page_url, remaining_entries):
    driver.get(page_url)
    wait = WebDriverWait(driver, 30)
    try:
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'a.ProductRowCard-module__cardTitleLink--YPqiC')))
    except TimeoutException:
        print("Error: Timeout while waiting for product elements to load.")
        return []

    product_elements = retry_on_stale_element(driver.find_elements, By.CSS_SELECTOR,
                                              'a.ProductRowCard-module__cardTitleLink--YPqiC')

    product_urls = []
    if product_elements:
        for product in product_elements[:remaining_entries]:
            url = product.get_attribute('href')
            if url:
                full_url = "https://www.teacherspayteachers.com" + url if not url.startswith("http") else url
                product_urls.append(full_url)
                print(f"Extracted URL: {full_url}")

    return product_urls

# Function to handle pagination and extract URLs from all pages
def extract_urls_from_all_pages(keyword, total_pages, max_entries):
    base_url = "https://www.teacherspayteachers.com/browse"
    all_urls = []
    driver = webdriver.Chrome()
    entries_extracted = 0

    try:
        for page in range(1, total_pages + 1):
            remaining_entries = max_entries - entries_extracted
            if remaining_entries <= 0:
                break

            page_url = f"{base_url}?order=Rating-Count&page={page}&search={keyword}"
            print(f"Processing page {page}: {page_url}")

            urls_from_page = extract_urls_from_page(driver, page_url, remaining_entries)
            all_urls.extend(urls_from_page)
            entries_extracted += len(urls_from_page)

    finally:
        driver.quit()

    return all_urls[:max_entries]

# Function to save URLs to a CSV file in URL_List folder
def save_urls_to_csv(urls, csv_file):
    os.makedirs('URL_List', exist_ok=True)
    file_path = os.path.join('URL_List', csv_file)

    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Product URL'])
        for url in urls:
            writer.writerow([url])
    print(f"URLs saved to {file_path}")

# Function to extract data from a single URL
def scrape_product_data(driver, url):
    driver.get(url)
    driver.implicitly_wait(10)

    # Extract grade level and category data
    grade_level = None
    categories = []

    # Extract grade level (e.g., "Not Grade Specific", "PreK - 5th")
    grade_element = retry_on_stale_element(driver.find_element, By.CSS_SELECTOR,
                                           '[data-testid="RebrandedContentText"] .NotLinkedSection span')
    if grade_element:
        grade_level = grade_element.text

    # Extract category links (e.g., "Life Skills", "Special Education")
    category_elements = retry_on_stale_element(driver.find_elements, By.CSS_SELECTOR,
                                               'div[data-testid="LabeledSectionContent"] a.Link-module__link--GFbUH')
    if category_elements:
        categories = [category.text for category in category_elements]

    # Combine grade level and categories into preview keywords
    preview_keywords = f"{grade_level} " + ", ".join(categories)

    # Extract additional data for CSV
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
        rating_value = rating_text.split(' ')[1]
        number_of_ratings = rating_text.split(' ')[-2]

    price = retry_on_stale_element(driver.find_element, By.XPATH, "//meta[@name='price']").get_attribute('content')

    return title, short_description, long_description, rating_value, number_of_ratings, price, preview_keywords, url

# Function to process URLs from a CSV and write data to another CSV in Info folder
def process_urls_from_csv(input_csv, output_csv, keyword):
    driver = webdriver.Chrome()
    os.makedirs('Info', exist_ok=True)
    output_file_path = os.path.join('Info', output_csv)

    try:
        with open(output_file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)
            writer.writerow(
                ['Title', 'Short Description', 'Long Description', 'Overall Rating', 'Number of Ratings',
                 'Price', 'Preview Keywords', 'URL', 'Search Keyword'])

            with open(input_csv, mode='r', encoding='utf-8') as url_file:
                reader = csv.reader(url_file)
                next(reader)
                count = 0
                for row in reader:
                    url = row[0]
                    count += 1
                    print(f"Processing URL {count}: {url}")
                    try:
                        product_data = scrape_product_data(driver, url)
                        writer.writerow(product_data + (keyword,))
                    except Exception as e:
                        print(f"Error processing {url}: {e}")
    finally:
        driver.quit()
    print(f"Data saved to {output_file_path}")

# Main script logic
if __name__ == "__main__":
    keywords_input = input("Enter search keywords separated by commas (e.g., 'behavior management, classroom rules'): ")
    keywords = [keyword.strip() for keyword in keywords_input.split(",")]

    max_entries_per_keyword = 0
    while not (1 <= max_entries_per_keyword <= 500):
        try:
            max_entries_per_keyword = int(input("Enter the number of entries to extract per keyword (1-500): "))
            if not (1 <= max_entries_per_keyword <= 500):
                print("Please enter a number between 1 and 500.")
        except ValueError:
            print("Invalid input. Please enter a valid number between 1 and 500.")

    items_per_page = 24

    for keyword in keywords:
        total_pages = (
                                  max_entries_per_keyword + items_per_page - 1) // items_per_page

        encoded_keyword = keyword.replace(" ", "%20")

        url_csv_file = f'{keyword.replace(" ", "_")}.csv'
        info_csv_file = f'{keyword.replace(" ", "_")}.csv'

        print(f"Extracting URLs for keyword: '{keyword}'")
        product_urls = extract_urls_from_all_pages(encoded_keyword, total_pages, max_entries_per_keyword)
        save_urls_to_csv(product_urls, url_csv_file)

        print(f"Processing URLs and saving data for keyword: '{keyword}'")
        process_urls_from_csv(os.path.join('URL_List', url_csv_file), info_csv_file, keyword)
