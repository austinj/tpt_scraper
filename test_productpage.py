import requests
from bs4 import BeautifulSoup

# Use a known product URL (replace with a real one you have)
product_url = "https://www.teacherspayteachers.com/Product/Bright-Vintage-Classroom-Decor-Bundle-Bright-Classroom-Theme-5070657" 

response = requests.get(product_url)
if response.status_code == 200:
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Extract the product title (here using the page title)
    title = soup.title.string if soup.title else "No title found"
    print("Title:", title)
    
    # Extract meta description (short description)
    meta_desc = soup.find("meta", {"name": "description"})
    short_description = meta_desc.get("content") if meta_desc else "No meta description found"
    print("Short Description:", short_description)
    
    # Extract long description (using a CSS class similar to your Selenium code)
    long_desc_elem = soup.find(class_="ProductDescriptionLayout__htmlDisplay")
    long_description = long_desc_elem.get_text(strip=True) if long_desc_elem else "No long description found"
    print("Long Description (snippet):", long_description[:100] + "...")
    
    # Extract rating element
    rating_elem = soup.select_one("span.StarRating-module__srOnly--FAzEA")
    rating = rating_elem.get_text(strip=True) if rating_elem else "No rating found"
    print("Rating:", rating)
    
    # Extract price from meta tag
    meta_price = soup.find("meta", {"name": "price"})
    price = meta_price.get("content") if meta_price else "No price found"
    print("Price:", price)
else:
    print(f"Failed to fetch the product page: {response.status_code}")
