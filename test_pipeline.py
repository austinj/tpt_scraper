"""
End-to-end pipeline test for the first 10 free products in coding.json.

Steps:
  1. Search  — grab first 10 product URLs from the first search page
  2. Scrape  — extract metadata for those 10 URLs
  3. Download — (requires tpt_storage.json) download files
  4. DB      — writes everything to test_pipeline.db for inspection

Run:
  .venv\\Scripts\\python.exe test_pipeline.py
  .venv\\Scripts\\python.exe test_pipeline.py --download        # include download step
  .venv\\Scripts\\python.exe test_pipeline.py --download --keep # keep downloaded files
"""
import argparse
import asyncio
import json
import logging
import re
import shutil
import sqlite3
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import aiohttp
import async_timeout
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CONFIG_FILE  = "configs/coding.json"
SESSION_FILE = "tpt_storage.json"
TEST_DB      = "test_pipeline.db"
LIMIT        = 10

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
}

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def init_db(db_path: str):
    con = sqlite3.connect(db_path)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS search_results (
            url TEXT PRIMARY KEY,
            price_option TEXT,
            discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS product_metadata (
            url TEXT PRIMARY KEY,
            product_id TEXT,
            title TEXT,
            short_description TEXT,
            long_description TEXT,
            product_price TEXT,
            rating_value TEXT,
            number_of_ratings TEXT,
            preview_keywords TEXT,
            author_name TEXT,
            author_store_url TEXT,
            author_follower_count INTEGER,
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS downloads (
            product_url TEXT PRIMARY KEY,
            file_path TEXT,
            file_size INTEGER,
            failure_reason TEXT,
            downloaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    con.commit()
    return con


def save_search_results(con: sqlite3.Connection, urls: list[str]):
    con.executemany(
        "INSERT OR IGNORE INTO search_results (url, price_option) VALUES (?, 'free')",
        [(u,) for u in urls]
    )
    con.commit()


def save_metadata(con: sqlite3.Connection, records: list[dict]):
    con.executemany("""
        INSERT OR REPLACE INTO product_metadata
            (url, product_id, title, short_description, long_description,
             product_price, rating_value,
             number_of_ratings, preview_keywords,
             author_name, author_store_url, author_follower_count)
        VALUES
            (:url, :product_id, :title, :short_description, :long_description,
             :product_price, :rating_value,
             :number_of_ratings, :preview_keywords,
             :author_name, :author_store_url, :author_follower_count)
    """, records)
    con.commit()


def save_download(con: sqlite3.Connection, product_url: str,
                  file_path: str | None, file_size: int | None,
                  failure_reason: str | None):
    con.execute("""
        INSERT OR REPLACE INTO downloads
            (product_url, file_path, file_size, failure_reason)
        VALUES (?, ?, ?, ?)
    """, (product_url, file_path, file_size, failure_reason))
    con.commit()


def print_db_summary(db_path: str):
    con = sqlite3.connect(db_path)
    print(f"\n{'='*70}")
    print(f"DB summary: {db_path}")
    print(f"{'='*70}")

    rows = con.execute("SELECT COUNT(*) FROM search_results").fetchone()[0]
    print(f"  search_results   : {rows} URLs")

    rows = con.execute("SELECT COUNT(*) FROM product_metadata").fetchone()[0]
    print(f"  product_metadata : {rows} records")

    dl_ok   = con.execute("SELECT COUNT(*) FROM downloads WHERE file_path IS NOT NULL").fetchone()[0]
    dl_fail = con.execute("SELECT COUNT(*) FROM downloads WHERE failure_reason IS NOT NULL").fetchone()[0]
    print(f"  downloads        : {dl_ok} succeeded, {dl_fail} failed")

    print(f"\n  {'product_id':<12} {'price':>6}  {'rating':>6}  {'ratings':>7}  {'followers':>9}  title")
    print(f"  {'-'*12} {'-'*6}  {'-'*6}  {'-'*7}  {'-'*9}  {'-'*40}")
    for r in con.execute("""
        SELECT product_id, product_price, rating_value, number_of_ratings,
               author_follower_count, title
        FROM product_metadata ORDER BY CAST(number_of_ratings AS INTEGER) DESC
    """):
        pid, price, rating, nratings, followers, title = r
        print(f"  {pid or '':<12} {price or '':>6}  {rating or '':>6}  {nratings or '':>7}  {followers or '':>9}  {(title or '')[:50]}")

    if dl_ok > 0:
        print(f"\n  Downloads:")
        for r in con.execute("SELECT product_url, file_path, file_size FROM downloads WHERE file_path IS NOT NULL"):
            url, fp, sz = r
            pid = re.search(r'-(\d+)$', url.rstrip('/'))
            pid = pid.group(1) if pid else '?'
            fname = Path(fp).name if fp else ''
            print(f"    [{pid}]  {fname}  ({sz:,} bytes)" if sz else f"    [{pid}]  {fname}")

    if dl_fail > 0:
        print(f"\n  Failures:")
        for r in con.execute("SELECT product_url, failure_reason FROM downloads WHERE failure_reason IS NOT NULL"):
            url, reason = r
            pid = re.search(r'-(\d+)$', url.rstrip('/'))
            pid = pid.group(1) if pid else '?'
            print(f"    [{pid}]  {reason}")

    con.close()
    print(f"\n  Open with: sqlite3 {db_path}  (or any SQLite viewer)")


# ---------------------------------------------------------------------------
# Helpers (copied from tpt_scraper.py so this script is standalone)
# ---------------------------------------------------------------------------

def build_page_url(resource_type, grade_level, subject, format_type,
                   price_option, supports, sort_order, page):
    parts = ["https://www.teacherspayteachers.com/browse"]
    if resource_type:  parts.append(resource_type)
    if grade_level:    parts.append(grade_level)
    if subject:        parts.append(subject)
    if format_type:    parts.append(format_type)
    if price_option:   parts.append(price_option)
    if supports:       parts.append(supports)
    base = "/".join(parts)
    qs = []
    if sort_order and sort_order != "Relevance":
        qs.append(f"order={sort_order}")
    if page > 1:
        qs.append(f"page={page}")
    return f"{base}?{'&'.join(qs)}" if qs else base


def extract_product_id(url):
    m = re.search(r"/Product/[^/]+-(\d+)(?:\?|$)", url)
    return m.group(1) if m else str(abs(hash(url)) % (10 ** 8))


def extract_text(el):
    if not el:
        return None
    text = re.sub(r"\s+", " ", el.get_text(separator=" ")).strip()
    return text or None


def extract_zip_in_place(zip_path: Path) -> bool:
    """Extract a zip file into its containing folder."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(zip_path.parent)
        return True
    except Exception as e:
        logging.warning(f"Failed to extract {zip_path.name}: {e}")
        return False


# ---------------------------------------------------------------------------
# Step 1: Search
# ---------------------------------------------------------------------------

async def step_search(config) -> list[str]:
    """Return the first LIMIT product URLs from the first search page."""
    grade   = config["grade_level"][0]
    subject = config["subject"][0]
    rtype   = config["resource_type"][0]
    fmt     = config["format"][0]
    price   = config["price_options"][0]
    sup     = config["supports"][0]
    sort    = config["sorting_methods"][0]

    url = build_page_url(rtype, grade, subject, fmt, price, sup, sort, 1)
    print(f"\n[Step 1: Search]  {url}")

    async with aiohttp.ClientSession() as session:
        async with async_timeout.timeout(30):
            async with session.get(url, headers=HEADERS) as resp:
                print(f"  HTTP {resp.status}")
                if resp.status != 200:
                    print("  FAIL: non-200 response")
                    return []
                html = await resp.text()

    soup  = BeautifulSoup(html, "lxml")
    links = soup.select("a.ProductRowCard-module__cardTitleLink--YPqiC")
    urls  = []
    for lk in links[:LIMIT]:
        href = lk.get("href", "")
        full = f"https://www.teacherspayteachers.com{href}" if not href.startswith("http") else href
        urls.append(full)

    print(f"  Found {len(links)} products on page — using first {len(urls)}")
    for u in urls:
        print(f"    {u}")
    return urls


# ---------------------------------------------------------------------------
# Step 2: Scrape metadata
# ---------------------------------------------------------------------------

async def scrape_one(session, url: str) -> dict | None:
    async with async_timeout.timeout(30):
        async with session.get(url, headers=HEADERS) as resp:
            if resp.status != 200:
                return None
            html = await resp.text()

    soup = BeautifulSoup(html, "lxml")

    title = soup.title.string if soup.title else None

    meta_desc = soup.find("meta", {"name": "description"})
    short_description = (
        meta_desc["content"] if meta_desc and meta_desc.has_attr("content") else None
    )

    long_desc_elem = soup.select_one('div[class*="htmlDisplay"]')
    long_description = None
    if long_desc_elem:
        text = re.sub(r"\s+", " ", long_desc_elem.get_text(separator=" ")).strip()
        long_description = text or None

    meta = lambda prop: (
        soup.find("meta", {"property": prop}) or {}
    ).get("content")

    rating_value     = meta("og:rating")
    number_of_ratings = meta("og:rating_count")
    product_price    = meta("product:price:amount")

    grade_elem = soup.find(attrs={"data-testid": "GradesLabel"})
    grade_level = None
    if grade_elem:
        grade_level = re.sub(r"^Mostly used with\s*", "",
                             grade_elem.get_text(strip=True)).strip() or None

    categories = [el.get_text(strip=True)
                  for el in soup.select('[data-testid^="Link_"]')]

    preview_keywords = (
        (grade_level or "")
        + (" " if grade_level and categories else "")
        + ", ".join(categories)
    ) or None

    author_name = None
    author_store_url = None
    al = soup.find(attrs={"data-testid": "authorAvatarLink"})
    if al:
        href = al.get("href", "")
        author_store_url = (
            f"https://www.teacherspayteachers.com{href}"
            if not href.startswith("http") else href
        )
        av = al.find(attrs={"data-testid": "authorAvatar"})
        if av:
            author_name = av.get("aria-label", "").strip() or None

    follower_count = None
    fc = soup.find("div", class_=re.compile(r"AboutAuthorRow-module__followContainer"))
    if fc:
        m = re.search(r"([\d,.]+)\s*([kKmM])?\s*Followers?",
                      fc.get_text(strip=True), re.IGNORECASE)
        if m:
            n = float(m.group(1).replace(",", ""))
            s = (m.group(2) or "").lower()
            if s == "k": n *= 1000
            elif s == "m": n *= 1_000_000
            follower_count = int(n)
        else:
            follower_count = 0

    return dict(
        url=url,
        product_id=extract_product_id(url),
        title=title,
        short_description=short_description,
        long_description=long_description,
        product_price=product_price,
        rating_value=rating_value,
        number_of_ratings=number_of_ratings,
        preview_keywords=preview_keywords,
        author_name=author_name,
        author_store_url=author_store_url,
        author_follower_count=follower_count,
    )


async def step_scrape(urls: list[str]) -> list[dict]:
    print(f"\n[Step 2: Scrape metadata]  {len(urls)} products")
    results = []
    async with aiohttp.ClientSession() as session:
        tasks = [scrape_one(session, u) for u in urls]
        raw = await asyncio.gather(*tasks, return_exceptions=True)

    fields = ["product_id", "title", "short_description", "long_description",
              "product_price", "rating_value", "number_of_ratings",
              "author_name", "author_follower_count", "preview_keywords"]

    failures = 0
    for url, r in zip(urls, raw):
        pid = extract_product_id(url)
        if isinstance(r, Exception) or r is None:
            print(f"  [{pid}] FAIL: {r}")
            failures += 1
            continue
        results.append(r)
        print(f"  [{pid}] OK")
        for f in fields:
            v = r.get(f)
            print(f"    {f:<22} {str(v)[:70] if v is not None else '(none)'}")

    print(f"\n  Scraped: {len(results)}/{len(urls)}  Failures: {failures}")
    return results


# ---------------------------------------------------------------------------
# Step 3: Download
# ---------------------------------------------------------------------------

class SessionExpiredError(Exception):
    pass


async def download_one(product_url: str, browser, downloads_dir: Path):
    free_url = product_url.replace("/Product/", "/FreeDownload/")
    product_id = extract_product_id(product_url)
    product_dir = downloads_dir / product_id
    product_dir.mkdir(parents=True, exist_ok=True)

    context = None
    try:
        context = await browser.new_context(
            storage_state=SESSION_FILE, accept_downloads=True
        )
        page = await context.new_page()
        page.on("dialog", lambda d: asyncio.ensure_future(d.dismiss()))

        try:
            async with page.expect_download(timeout=15000) as dl_info:
                await page.goto(free_url, wait_until="domcontentloaded", timeout=30000)
            download = await dl_info.value
        except Exception:
            current_url = page.url
            if any(t in current_url.lower() for t in
                   ("login", "signin", "sign-in", "request-authorization")):
                raise SessionExpiredError(
                    "Session expired — re-run create_session.py"
                )
            btn = page.locator(
                '[data-testid="download-button-cta"], [data-testid="download-button"]'
            ).first
            try:
                visible = await btn.is_visible(timeout=3000)
            except Exception:
                visible = False
            if visible:
                async with page.expect_download(timeout=30000) as dl_info:
                    await btn.click()
                download = await dl_info.value
            else:
                return None, f"no_download_triggered (landed on {current_url[:80]})"

        save_path = product_dir / download.suggested_filename
        await download.save_as(save_path)

        if not save_path.exists() or save_path.stat().st_size == 0:
            return None, "empty_file"

        return save_path, None

    except SessionExpiredError:
        raise
    except Exception as e:
        return None, str(e)[:120]
    finally:
        if context:
            try:
                await context.close()
            except Exception:
                pass


async def step_download(urls: list[str], downloads_dir: Path, con: sqlite3.Connection):
    print(f"\n[Step 3: Download]  {len(urls)} products  ->  {downloads_dir}")

    success, failures = 0, 0
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        print("  Browser launched")
        try:
            for url in urls:
                pid = extract_product_id(url)
                try:
                    path, err = await download_one(url, browser, downloads_dir)
                except SessionExpiredError as e:
                    print(f"\n  ABORT: {e}")
                    return success, failures
                if path:
                    size = path.stat().st_size
                    print(f"  [{pid}] OK  {path.name}  ({size:,} bytes)")
                    save_download(con, url, str(path), size, None)
                    if path.suffix.lower() == ".zip":
                        if extract_zip_in_place(path):
                            extracted = [f.name for f in path.parent.iterdir() if f != path]
                            print(f"    Extracted: {', '.join(extracted[:5])}")
                    success += 1
                else:
                    print(f"  [{pid}] FAIL  {err}")
                    save_download(con, url, None, None, err)
                    failures += 1
        finally:
            await browser.close()
            print("  Browser closed")

    return success, failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(run_download: bool, keep: bool):
    # Fresh test DB each run — skip wipe if file is locked (e.g. open in a viewer)
    db_path = TEST_DB
    try:
        Path(db_path).unlink(missing_ok=True)
    except PermissionError:
        print(f"NOTE: {db_path} is open in another process — appending to existing DB")
    con = init_db(db_path)
    with open(CONFIG_FILE) as f:
        config = json.load(f)

    print(f"Config: {CONFIG_FILE}")
    print(f"  price_options : {config['price_options']}")
    print(f"  subject       : {config['subject']}")
    print(f"  Limit         : {LIMIT} products")

    t0 = time.time()
    failures = []

    # Step 1
    urls = await step_search(config)
    if not urls:
        print("\nFAIL: no URLs found")
        sys.exit(1)
    save_search_results(con, urls)

    # Step 2
    metadata = await step_scrape(urls)
    if len(metadata) < len(urls):
        failures.append(f"scrape: {len(urls) - len(metadata)} failed")
    save_metadata(con, metadata)

    # Step 3
    if run_download:
        if not Path(SESSION_FILE).exists():
            print(f"\n[Step 3: Download]  SKIP — {SESSION_FILE} not found")
            print("  Run create_session.py first, then re-run with --download")
        else:
            dl_dir = Path(f"downloads_coding_test") if keep else Path(tempfile.mkdtemp(prefix="tpt_pipeline_test_"))
            if keep:
                dl_dir.mkdir(exist_ok=True)
                print(f"\n[Step 3: Download]  keeping files in {dl_dir}")
            try:
                ok, fail = await step_download(urls, dl_dir, con)
                if fail:
                    failures.append(f"download: {fail} failed")
            finally:
                if not keep:
                    shutil.rmtree(dl_dir, ignore_errors=True)
                    print(f"  Temp download folder removed")
    else:
        print(f"\n[Step 3: Download]  SKIP — pass --download to include")

    elapsed = time.time() - t0
    con.close()
    print_db_summary(db_path)
    print(f"\nElapsed: {elapsed:.1f}s")
    if failures:
        print(f"FAIL — {failures}")
        sys.exit(1)
    else:
        print("PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pipeline test for first 10 products")
    ap.add_argument("--download", action="store_true",
                    help="Also run the download step (requires tpt_storage.json)")
    ap.add_argument("--keep", action="store_true",
                    help="Keep downloaded files in downloads_coding_test/ instead of a temp folder")
    args = ap.parse_args()
    asyncio.run(main(args.download, args.keep))
