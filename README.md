# TeachersPayTeachers Web Scraper

This is a web scraper for [TeachersPayTeachers.com](https://www.teacherspayteachers.com), built using Selenium. The script allows users to input a **search keyword**, and then it scrapes the first **300 results** from the site. The script captures essential information about each resource, including:

- **Name of Resource**
- **Overall Rating**
- **Number of Ratings**
- **Description from Seller**
- **Keywords**
- **Grade Levels**
- **Price**

The collected data is then saved to a CSV file.

This web scraper was created as part of a paid project for educuational research under Lydia Beahm, a professor in the Department of Education and Human Development at Clemson University.

---

## Features

- **User Input**: The script prompts the user to enter a search keyword.
- **Pagination**: The script handles pagination to collect up to the first 300 results (12 pages with 25 results per page).
- **Data Capturing**: Captures detailed information such as ratings, description, price, and more for each resource.
- **CSV Export**: Stores all extracted information in a well-structured CSV file for easy access and analysis.

---

## Requirements

### Prerequisites

Before running the script, ensure you have the following installed:

1. **Python 3.x**: You can download it from [Python's official website](https://www.python.org/downloads/).
2. **Selenium**: Install it via pip:
    ```bash
    pip install selenium
    ```
3. **Chrome WebDriver**: Download the appropriate version of Chrome WebDriver for your system from [here](https://sites.google.com/a/chromium.org/chromedriver/downloads), and ensure it is accessible in your system's PATH.

### Additional Libraries

- **CSV**: Used for reading and writing to CSV files (part of Python's standard library).
- **Selenium WebDriver**: Used to control the web browser.

---

## How to Run

1. **Clone the repository**:
    ```bash
    git clone https://github.com/adamruns/teachers4teachers-scraper.git
    cd teachers4teachers-scraper
    ```

2. **Install dependencies** (if not done already):
    ```bash
    pip install selenium
    ```

3. **Download Chrome WebDriver** and ensure it's in your system's PATH.

4. **Run the script**:
    ```bash
    python extract_info.py
    ```

5. **Input the Search Keyword** when prompted:
    - Example: `classroom management`
    - The scraper will then fetch the first 300 results related to this keyword.

6. **Wait for the script to complete**: It will scrape all the relevant data and save it in a CSV file (`product_info_with_grades_ratings_price.csv`).

---

## Captured Information

The script captures the following information for each resource:

- **Name of Resource**: The title of the educational resource.
- **Overall Rating**: The average rating out of 5.
- **Number of Ratings**: The total number of ratings the resource has received.
- **Description from Seller**: The detailed description provided by the seller.
- **Keywords**: Any tags or keywords associated with the resource.
- **Grade Levels**: The grade levels (e.g., PreK, 1st Grade) for which the resource is suitable.
- **Price**: The listed price of the resource.

---

## Output

The script generates a CSV file named `product_info_with_grades_ratings_price.csv` with the following columns:

| Title | Short Description | Long Description | Grades Used | Overall Rating | Number of Ratings | Price | URL | Keyword |
|-------|-------------------|------------------|-------------|----------------|-------------------|-------|-----|---------|
| ...   | ...               | ...              | ...         | ...            | ...               | ...   | ... | ...     |

---

## Customization

You can modify the script to:
- **Change the number of pages** scraped by adjusting the `total_pages` variable.
- **Adjust search parameters** such as filtering by grade level, subject, or resource type.

---

## Troubleshooting

### Common Issues

1. **Stale Element Reference**: If you encounter a "Stale Element Reference" error, it means the page updated dynamically while the script was trying to access a specific element. The script handles some retries to mitigate this, but you can further increase the waiting time between page loads.
2. **WebDriver Issues**: Ensure that your version of **Chrome WebDriver** matches your installed version of Chrome. You can check this by going to `chrome://settings/help` in your Chrome browser and downloading the appropriate WebDriver from [here](https://sites.google.com/a/chromium.org/chromedriver/downloads).

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
