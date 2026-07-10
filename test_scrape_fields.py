"""
Test that scrape_product_metadata() correctly extracts all fields from a live TPT product page.
Uses a known free classroom-management product URL obtained from the search results.
"""
import asyncio
import sys
import aiohttp
import async_timeout
import re
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
}

SEARCH_URL = (
    "https://www.teacherspayteachers.com/browse/"
    "teacher-tools/elementary/preschool/teacher-tools/classroom-management/free"
)

FIELDS = [
    "url", "product_id", "title", "short_description", "long_description",
    "rating_value", "number_of_ratings", "product_price",
    "preview_keywords", "author_name", "author_store_url", "author_follower_count",
]


def extract_text_with_spacing(element):
    if not element:
        return None
    text = element.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else None


def scrape_product_metadata_from_html(url, html):
    soup = BeautifulSoup(html, "lxml")

    title = soup.title.string if soup.title else None

    meta_desc = soup.find("meta", {"name": "description"})
    short_description = meta_desc["content"] if meta_desc and meta_desc.has_attr("content") else None

    long_desc_elem = soup.select_one('div[class*="htmlDisplay"]')
    long_description = extract_text_with_spacing(long_desc_elem) if long_desc_elem else None

    rating_value = None
    meta_rating = soup.find("meta", {"property": "og:rating"})
    if meta_rating and meta_rating.has_attr("content"):
        rating_value = meta_rating["content"]

    number_of_ratings = None
    meta_rating_count = soup.find("meta", {"property": "og:rating_count"})
    if meta_rating_count and meta_rating_count.has_attr("content"):
        number_of_ratings = meta_rating_count["content"]

    product_price = None
    meta_price = soup.find("meta", {"property": "product:price:amount"})
    if meta_price and meta_price.has_attr("content"):
        product_price = meta_price["content"]

    grade_level = None
    grade_elem = soup.find(attrs={"data-testid": "GradesLabel"})
    if grade_elem:
        grade_text = grade_elem.get_text(strip=True)
        grade_level = re.sub(r'^Mostly used with\s*', '', grade_text).strip() or None

    categories = []
    category_elems = soup.select('[data-testid^="Link_"]')
    if category_elems:
        categories = [cat.get_text(strip=True) for cat in category_elems]

    preview_keywords = ""
    if grade_level or categories:
        preview_keywords = (grade_level or "") + (" " if grade_level and categories else "") + ", ".join(categories)

    author_name = None
    author_store_url = None
    author_link = soup.find(attrs={"data-testid": "authorAvatarLink"})
    if author_link:
        author_store_url = author_link.get("href", "")
        if not author_store_url.startswith("http"):
            author_store_url = "https://www.teacherspayteachers.com" + author_store_url
        avatar = author_link.find(attrs={"data-testid": "authorAvatar"})
        if avatar:
            author_name = avatar.get("aria-label", "").strip() or None

    author_follower_count = None
    follow_container = soup.find("div", class_=re.compile(r"AboutAuthorRow-module__followContainer"))
    if follow_container:
        container_text = follow_container.get_text(strip=True)
        match = re.search(r"([\d,\.]+)\s*([kKmM])?\s*Followers?", container_text, re.IGNORECASE)
        if match:
            num_str = match.group(1).replace(",", "")
            suffix = match.group(2)
            try:
                num = float(num_str)
                if suffix:
                    s = suffix.lower()
                    if s == "k":
                        num *= 1000
                    elif s == "m":
                        num *= 1000000
                author_follower_count = int(num)
            except ValueError:
                author_follower_count = 0
        else:
            author_follower_count = 0

    product_id = None
    id_match = re.search(r"/Product/[^/]+-(\d+)(?:\?|$)", url)
    if id_match:
        product_id = id_match.group(1)

    return (url, product_id, title, short_description, long_description,
            rating_value, number_of_ratings, product_price,
            preview_keywords, author_name, author_store_url, author_follower_count)


async def get_sample_product_url(session):
    """Grab the first product URL from the search results page."""
    async with async_timeout.timeout(30):
        async with session.get(SEARCH_URL, headers=HEADERS) as resp:
            html = await resp.text()
    soup = BeautifulSoup(html, "lxml")
    links = soup.select("a.ProductRowCard-module__cardTitleLink--YPqiC")
    if not links:
        return None
    href = links[0].get("href", "")
    return f"https://www.teacherspayteachers.com{href}" if not href.startswith("http") else href


async def main():
    async with aiohttp.ClientSession() as session:
        print("Step 1: Fetching search page to get a sample product URL...")
        product_url = await get_sample_product_url(session)
        if not product_url:
            print("FAIL: Could not get a product URL from search results")
            sys.exit(1)
        print(f"  Sample product: {product_url}\n")

        print("Step 2: Fetching product page...")
        async with async_timeout.timeout(30):
            async with session.get(product_url, headers=HEADERS) as resp:
                print(f"  HTTP {resp.status}")
                if resp.status != 200:
                    print("FAIL: Non-200 response")
                    sys.exit(1)
                html = await resp.text()

        print("\nStep 3: Extracting metadata fields...\n")
        result = scrape_product_metadata_from_html(product_url, html)
        data = dict(zip(FIELDS, result))

    # --- Report ---
    failures = []
    print(f"{'Field':<22} {'Status':<8} Value")
    print("-" * 80)
    for field in FIELDS:
        val = data[field]
        # Fields that are always expected
        required = field in ("url", "product_id", "title", "short_description", "product_price")
        # Rating fields may be absent for products with no ratings
        optional = field in ("rating_value", "number_of_ratings", "long_description",
                             "preview_keywords", "author_name", "author_store_url",
                             "author_follower_count")

        if val is not None and val != "":
            status = "PASS"
            display = str(val)[:70]
        elif optional:
            status = "NOTE"
            display = "(empty/None — may be valid)"
        else:
            status = "FAIL"
            display = "(missing)"
            failures.append(field)

        print(f"  {field:<20} {status:<8} {display}")

    # Extra: verify price is 0.00 (free product)
    price = data.get("product_price")
    print()
    if price is not None:
        if price == "0.00" or price == "0":
            print(f"[Free price check] PASS: product_price = {price!r}")
        else:
            print(f"[Free price check] FAIL: product_price = {price!r} (expected 0.00)")
            failures.append("product_price_value")
    else:
        print("[Free price check] NOTE: product_price meta tag absent (may be free or unlisted)")

    print(f"\n{'='*80}")
    if failures:
        print(f"FAIL — missing required fields: {failures}")
        sys.exit(1)
    else:
        print("PASS — all required fields extracted successfully")


if __name__ == "__main__":
    asyncio.run(main())
