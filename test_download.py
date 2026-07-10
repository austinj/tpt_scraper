"""
Test the download workflow for coding.json:
  1. URL transform /Product/ -> /FreeDownload/ is correct
  2. FreeDownload URL responds (not 404/gone)
  3. If tpt_storage.json exists: Playwright download actually triggers and produces a non-empty file
"""
import asyncio
import re
import sys
import tempfile
import shutil
from pathlib import Path

import aiohttp
import async_timeout
from bs4 import BeautifulSoup

SEARCH_URL = (
    "https://www.teacherspayteachers.com/browse/"
    "teacher-tools/elementary/preschool/teacher-tools/classroom-management/free"
)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
}
SESSION_FILE = "tpt_storage.json"


def make_free_download_url(product_url: str) -> str:
    return product_url.replace("/Product/", "/FreeDownload/")


def extract_product_id(url: str) -> str:
    match = re.search(r'/Product/[^/]+-(\d+)(?:\?|$)', url)
    if match:
        return match.group(1)
    return str(abs(hash(url)) % (10 ** 8))


async def get_sample_product_url(session) -> str | None:
    async with async_timeout.timeout(30):
        async with session.get(SEARCH_URL, headers=HEADERS) as resp:
            html = await resp.text()
    soup = BeautifulSoup(html, "lxml")
    links = soup.select("a.ProductRowCard-module__cardTitleLink--YPqiC")
    if not links:
        return None
    href = links[0].get("href", "")
    return f"https://www.teacherspayteachers.com{href}" if not href.startswith("http") else href


async def test_url_transform(product_url: str) -> bool:
    """Test 1: URL transform produces expected /FreeDownload/ path."""
    free_url = make_free_download_url(product_url)
    expected_pattern = r"https://www\.teacherspayteachers\.com/FreeDownload/"
    ok = bool(re.match(expected_pattern, free_url)) and "/Product/" not in free_url
    print(f"\n[Test 1: URL transform]")
    print(f"  Product URL : {product_url}")
    print(f"  Download URL: {free_url}")
    print(f"  {'PASS' if ok else 'FAIL'}: /FreeDownload/ substitution")
    return ok, free_url


async def test_url_reachable(session, free_url: str) -> bool:
    """Test 2: FreeDownload URL responds — not 404/410. 
    Without auth we expect a redirect to login (3xx) or the download page (200).
    404 would mean the URL pattern is broken."""
    print(f"\n[Test 2: FreeDownload URL reachable]")
    try:
        async with async_timeout.timeout(20):
            async with session.get(free_url, headers=HEADERS, allow_redirects=False) as resp:
                status = resp.status
                location = resp.headers.get("Location", "")
        
        if status in (200, 301, 302, 303, 307, 308):
            print(f"  PASS: HTTP {status}" + (f" -> {location[:80]}" if location else ""))
            return True
        elif status == 404:
            print(f"  FAIL: HTTP 404 — /FreeDownload/ URL pattern may be broken")
            return False
        elif status == 410:
            print(f"  FAIL: HTTP 410 Gone — product may no longer exist")
            return False
        else:
            print(f"  NOTE: HTTP {status} — unexpected but not necessarily broken")
            return True
    except Exception as e:
        print(f"  FAIL: Request error — {e}")
        return False


async def test_playwright_download(product_url: str, free_url: str) -> bool:
    """Test 3: Playwright actually triggers a download (requires tpt_storage.json)."""
    from playwright.async_api import async_playwright

    print(f"\n[Test 3: Playwright download]")
    tmp_dir = Path(tempfile.mkdtemp(prefix="tpt_test_download_"))
    print(f"  Temp dir: {tmp_dir}")
    product_id = extract_product_id(product_url)

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                storage_state=SESSION_FILE,
                accept_downloads=True
            )
            page = await context.new_page()

            print(f"  Navigating to: {free_url}")
            try:
                async with page.expect_download(timeout=30000) as download_info:
                    await page.goto(free_url, wait_until="domcontentloaded", timeout=30000)
                download = await download_info.value
            except Exception as e:
                # If expect_download times out it may mean the page requires
                # interaction (e.g. a "Download" button click) before the file
                # is served — check what appeared on the page
                content = await page.content()
                await context.close()
                await browser.close()

                if "sign" in content.lower() or "log in" in content.lower():
                    print(f"  FAIL: Redirected to login — session may be expired")
                    print(f"  Re-run create_session.py to refresh tpt_storage.json")
                else:
                    # Check for a download button
                    soup = BeautifulSoup(content, "lxml")
                    dl_btn = soup.find(attrs={"data-testid": "download-button"})
                    if dl_btn:
                        print(f"  NOTE: Page loaded but download requires a button click")
                        print(f"  The /FreeDownload/ URL pattern is valid; download triggering may need updating")
                    else:
                        print(f"  FAIL: Download did not trigger — {e}")
                return False

            filename = download.suggested_filename
            save_path = tmp_dir / filename
            await download.save_as(save_path)
            await context.close()
            await browser.close()

        if save_path.exists() and save_path.stat().st_size > 0:
            size = save_path.stat().st_size
            print(f"  PASS: Downloaded '{filename}' ({size:,} bytes)")
            return True
        else:
            print(f"  FAIL: File missing or empty after download")
            return False

    except Exception as e:
        print(f"  FAIL: {e}")
        return False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"  Temp dir cleaned up")


async def main():
    failures = []

    async with aiohttp.ClientSession() as session:
        print("Fetching a sample free product URL from search results...")
        product_url = await get_sample_product_url(session)
        if not product_url:
            print("FAIL: Could not get a product URL — search selector may be broken")
            sys.exit(1)
        print(f"Sample product: {product_url}")

        # Test 1
        ok1, free_url = await test_url_transform(product_url)
        if not ok1:
            failures.append("url_transform")

        # Test 2
        ok2 = await test_url_reachable(session, free_url)
        if not ok2:
            failures.append("url_reachable")

    # Test 3
    session_path = Path(SESSION_FILE)
    if session_path.exists():
        ok3 = await test_playwright_download(product_url, free_url)
        if not ok3:
            failures.append("playwright_download")
    else:
        print(f"\n[Test 3: Playwright download]")
        print(f"  SKIP: {SESSION_FILE} not found — run create_session.py to enable this test")

    print(f"\n{'='*70}")
    if failures:
        print(f"FAIL — {failures}")
        sys.exit(1)
    else:
        skipped = " (Test 3 skipped — no session file)" if not session_path.exists() else ""
        print(f"PASS{skipped}")


if __name__ == "__main__":
    asyncio.run(main())
