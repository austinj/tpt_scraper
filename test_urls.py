import requests
from bs4 import BeautifulSoup

# URL for the search results page
search_url = "https://www.teacherspayteachers.com/browse/teacher-tools/classroom-management"

response = requests.get(search_url)
if response.status_code == 200:
    soup = BeautifulSoup(response.text, "html.parser")
    # Using the same CSS selector as in Selenium
    product_links = soup.select("a.ProductRowCard-module__cardTitleLink--YPqiC")
    if product_links:
        print(f"Found {len(product_links)} product URLs:")
        for link in product_links[:5]:  # print first 5 for example
            href = link.get("href")
            # Prepend domain if needed:
            full_url = f"https://www.teacherspayteachers.com{href}" if not href.startswith("http") else href
            print(full_url)
    else:
        print("No product URL elements found in search results. They might be loaded via JavaScript.")
else:
    print(f"Failed to fetch the search page: {response.status_code}")
