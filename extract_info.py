import csv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

# Function to extract data from a single URL
def scrape_product_data(driver, url):
    driver.get(url)
    driver.implicitly_wait(10)  # Wait for 10 seconds for the elements to load

    # Extract grade data from the bar chart
    grade_data = {}
    grade_labels = driver.find_elements(By.CSS_SELECTOR, '.BarChart__labels--text')
    grade_values = driver.find_elements(By.CSS_SELECTOR, '[data-testid="bar-container-text"]')

    # Save the grade data as a dictionary
    for label, value in zip(grade_labels, grade_values):
        grade_data[label.text] = value.text

    # Extract the page title
    title = driver.title

    # Extract short description from the meta tag
    short_description = driver.find_element(By.XPATH, "//meta[@name='description']").get_attribute('content')

    # Extract long description
    long_description = driver.find_element(By.CLASS_NAME, 'ProductDescriptionLayout__htmlDisplay').text

    # Extract rating and number of ratings
    rating_value = None
    number_of_ratings = None
    try:
        rating_section = driver.find_element(By.CSS_SELECTOR, 'span.StarRating-module__srOnly--FAzEA')
        if rating_section:
            rating_text = rating_section.get_attribute('innerText')
            rating_value = rating_text.split(' ')[1]  # Get the rating value
            number_of_ratings = rating_text.split(' ')[-2]  # Get the number of ratings
    except Exception as e:
        print(f"Error extracting rating: {e}")

    # Extract price data
    price = None
    try:
        price = driver.find_element(By.XPATH, "//meta[@name='price']").get_attribute('content')
    except Exception as e:
        print(f"Error extracting price: {e}")

    # Formatting the grade data as a string
    grades_formatted = ', '.join([f"{key}: {value}" for key, value in grade_data.items()])

    # Return all the extracted data as a tuple (without the keyword yet)
    return title, short_description, long_description, grades_formatted, rating_value, number_of_ratings, price, url

# Function to read URLs from a CSV and write data to another CSV
def process_urls_from_csv(input_csv, output_csv, keyword):
    # Initialize WebDriver
    driver = webdriver.Chrome()

    try:
        # Open the output CSV file for writing
        with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)
            # Writing the header (with the keyword column)
            writer.writerow(['Title', 'Short Description', 'Long Description', 'Grades Used', 'Overall Rating', 'Number of Ratings', 'Price', 'URL', 'Keyword'])

            # Read the URLs from the input CSV
            with open(input_csv, mode='r', encoding='utf-8') as url_file:
                reader = csv.reader(url_file)
                next(reader)  # Skip the header

                count = 0  # Initialize a counter
                for row in reader:
                    url = row[0]  # Assuming each URL is in the first column of the CSV
                    count += 1  # Increment the counter for each URL
                    print(f"Processing URL {count}: {url}")  # Print the count and the URL

                    # Scrape data from the current URL
                    try:
                        product_data = scrape_product_data(driver, url)
                        # Append the keyword to the product data tuple and write to the CSV
                        writer.writerow(product_data + (keyword,))
                    except Exception as e:
                        print(f"Error processing {url}: {e}")

    finally:
        # Close the browser once all URLs are processed
        driver.quit()

# Main script logic
if __name__ == "__main__":
    input_csv = 'product_urls.csv'  # CSV file containing the URLs
    output_csv = 'product_info_with_grades_ratings_price.csv'  # CSV file to save the scraped data
    keyword = 'behavior management'  # The keyword to be added to every row

    # Process all URLs from the input CSV and save data to output CSV
    process_urls_from_csv(input_csv, output_csv, keyword)
