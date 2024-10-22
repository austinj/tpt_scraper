import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# Function to extract URLs from a single page
def extract_urls_from_page(driver, page_url):
    driver.get(page_url)

    # Wait for the products to load
    wait = WebDriverWait(driver, 20)
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '.ProductRowLayout')))

    # Find all product elements
    product_elements = driver.find_elements(By.CSS_SELECTOR, '.ProductRowCard-module__linkArea--aCqXC')

    # Extract URLs
    product_urls = []
    for product in product_elements:
        url = product.get_attribute('href')
        if url:
            full_url = "https://www.teacherspayteachers.com" + url if not url.startswith("http") else url
            product_urls.append(full_url)
            print(f"Extracted URL: {full_url}")  # Debugging: Print each URL extracted

    return product_urls


# Function to handle pagination and extract URLs from all pages
def extract_urls_from_all_pages(keyword, total_pages):
    base_url = "https://www.teacherspayteachers.com/browse"
    all_urls = []

    # Initialize Selenium WebDriver
    driver = webdriver.Chrome()

    try:
        # Iterate through the pages
        for page in range(1, total_pages + 1):
            # Construct the URL for the current page
            page_url = f"{base_url}?page={page}&search={keyword}"
            print(f"Processing page {page}: {page_url}")  # Debugging: Print the page URL being processed

            # Extract URLs from the current page
            urls_from_page = extract_urls_from_page(driver, page_url)
            all_urls.extend(urls_from_page)

    finally:
        # Close the browser once all pages are processed
        driver.quit()

    return all_urls


# Function to save URLs to a CSV file
def save_urls_to_csv(urls, csv_file):
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Product URL'])
        for url in urls:
            writer.writerow([url])
    print(f"URLs saved to {csv_file}")


# Main script logic
if __name__ == "__main__":
    # Ask user to input the keyword
    keyword_input = input("Enter the search keyword (e.g., 'behavior management'): ")
    # URL encode the keyword (replace spaces with %20 for URLs)
    keyword = keyword_input.replace(" ", "%20")

    total_pages = 13  # Total pages to process (as per the example)
    csv_file = 'product_urls.csv'  # CSV file to save URLs

    # Extract URLs from all pages
    product_urls = extract_urls_from_all_pages(keyword, total_pages)

    # Save the extracted URLs to a CSV file
    save_urls_to_csv(product_urls, csv_file)
