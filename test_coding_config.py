"""
Quick test to verify coding.json config still works with current TPT page structure
and that results are limited to free materials.
"""
import asyncio
import json
import sys
import aiohttp
import async_timeout
from bs4 import BeautifulSoup

CONFIG_FILE = "configs/coding.json"

def build_page_url(resource_type, grade_level, subject, format_type, price_option, supports, sort_order, page):
    url_parts = ["https://www.teacherspayteachers.com/browse"]
    if resource_type: url_parts.append(resource_type)
    if grade_level: url_parts.append(grade_level)
    if subject: url_parts.append(subject)
    if format_type: url_parts.append(format_type)
    if price_option: url_parts.append(price_option)
    if supports: url_parts.append(supports)
    base_url = "/".join(url_parts)
    query_params = []
    if sort_order and sort_order != "Relevance":
        query_params.append(f"order={sort_order}")
    if page > 1:
        query_params.append(f"page={page}")
    return f"{base_url}?{'&'.join(query_params)}" if query_params else base_url


async def fetch(session, url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
    }
    async with async_timeout.timeout(30):
        async with session.get(url, headers=headers) as resp:
            print(f"  HTTP {resp.status}: {url}")
            if resp.status == 200:
                return await resp.text()
            return None


async def test_search_page(config, url):
    """Fetch one search page and verify selector + free-only results."""
    async with aiohttp.ClientSession() as session:
        html = await fetch(session, url)

    if not html:
        print("FAIL: Could not fetch page (non-200 response)")
        return False

    soup = BeautifulSoup(html, "lxml")

    # --- Test 1: product card selector ---
    SELECTOR = "a.ProductRowCard-module__cardTitleLink--YPqiC"
    products = soup.select(SELECTOR)
    print(f"\n[Selector test] '{SELECTOR}'")
    if products:
        print(f"  PASS: found {len(products)} product links")
        for p in products[:3]:
            print(f"    {p.get_text(strip=True)[:80]}  ->  {p.get('href','')[:60]}")
    else:
        print("  FAIL: selector matched 0 elements — may need updating")
        # Dump candidate <a> tags so we can find the new class
        candidates = soup.select("a[class*='ProductRowCard']")
        if candidates:
            print(f"  Candidates with 'ProductRowCard' in class ({len(candidates)} found):")
            for c in candidates[:5]:
                print(f"    class={c.get('class')}  href={c.get('href','')[:60]}")
        else:
            print("  No <a> tags with 'ProductRowCard' in class found either.")
        return False

    # --- Test 2: verify price_option=free is in the URL path ---
    price_in_url = "free" in url
    print(f"\n[Free filter test] 'free' in URL path: {'PASS' if price_in_url else 'FAIL'}")
    print(f"  URL: {url}")

    # --- Test 3: verify none of the scraped ProductRowCard items are paid ---
    # Note: TPT also renders 'ProductGridCard' items (sponsored/related) with paid prices
    # on free-filtered pages, but those use a different selector and won't be captured.
    paid_row_cards = []
    for rc in products:
        container = rc
        for _ in range(6):
            container = container.parent
            if container is None:
                break
            price = container.select_one("[class*='ProductPrice-module__price']")
            if price:
                text = price.get_text(strip=True)
                if not (text.startswith("Free") or text.startswith("FREE") or text == "$0.00"):
                    paid_row_cards.append((rc.get("href", ""), text))
                break

    grid_cards = soup.select("[class*='ProductGridCard-module__linkArea']")
    paid_grid = sum(
        1 for gc in grid_cards
        if (p := gc.select_one("[class*='ProductPrice-module__price']"))
        and not p.get_text(strip=True).startswith("Free")
        and not p.get_text(strip=True).startswith("FREE")
    )

    if paid_row_cards:
        print(f"\n[Price check] FAIL: {len(paid_row_cards)} ProductRowCard items are paid (scraper would collect these):")
        for href, price in paid_row_cards[:5]:
            print(f"    {price!r}  {href[:60]}")
    else:
        print(f"\n[Price check] PASS: all {len(products)} ProductRowCard items are free")
    if paid_grid:
        print(f"  NOTE: {paid_grid} paid ProductGridCard (sponsored) items on page — scraper selector does NOT capture these")

    if paid_row_cards:
        return False

    # --- Test 4: config price_options value ---
    price_options = config.get("price_options", [])
    if price_options == ["free"]:
        print(f"\n[Config check] PASS: price_options = {price_options}")
    else:
        print(f"\n[Config check] FAIL: price_options = {price_options} (expected [\"free\"])")
        return False

    return True


async def main():
    with open(CONFIG_FILE) as f:
        config = json.load(f)

    print(f"Loaded config: {CONFIG_FILE}")
    print(f"  resource_type : {config.get('resource_type')}")
    print(f"  subject       : {config.get('subject')}")
    print(f"  price_options : {config.get('price_options')}")
    print(f"  grade_levels  : {len(config.get('grade_level', []))} entries")
    print(f"  sorting_methods: {config.get('sorting_methods')}")

    # Build one test URL (first grade level, first sort, page 1)
    grade = config["grade_level"][0]
    subject = config["subject"][0]
    resource_type = config["resource_type"][0]
    format_type = config["format"][0]
    price_option = config["price_options"][0]
    supports = config["supports"][0]
    sort_order = config["sorting_methods"][0]

    url = build_page_url(resource_type, grade, subject, format_type, price_option, supports, sort_order, 1)

    print(f"\nTest URL: {url}\n{'='*70}")

    passed = await test_search_page(config, url)

    print(f"\n{'='*70}")
    print(f"Overall: {'PASS' if passed else 'FAIL'}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
