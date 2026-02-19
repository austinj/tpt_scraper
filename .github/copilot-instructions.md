# Copilot Instructions — TPT Scraper

## Project Overview

This is a Python async scraper for Teachers Pay Teachers (TPT). It has four sequential workflows: **search** → **scrape** → **deepscrape** → **download**, each operating on a named configuration. Each config gets its own SQLite database.

## Architecture

- **`tpt_scraper.py`** — Main entry point. All workflows, DB setup, CLI parsing. ~1300 lines, single-file by design.
- **`config_manager.py`** — Manages named configs in `configs/` with a `registry.json` mapping config names to DB files.
- **`create_session.py`** — Generates `tpt_storage.json` (Playwright auth state for TPT).
- **`test_tpt_scraper.py`** — Pytest test suite (`asyncio_mode = auto`).

## Key Technical Details

### Async Stack
- **aiohttp** + **aiohttp_client_cache** for HTTP requests (search + scrape)
- **aiosqlite** for async SQLite with WAL mode
- **Playwright** (async, Chromium) for deep scrape and downloads
- **BeautifulSoup** + **lxml** for HTML parsing

### Database
- SQLite with WAL mode, one DB per config: `scrape_cache_<config>.db`
- Tables: `search_results`, `product_metadata`, `downloads`, `download_queue`
- `oldest_review_date` uses sentinel values: `'no_reviews'`, `'parse_failed'` (strings, not NULL) for resume support
- All workflows are resumable — they skip rows that already have data

### Deep Scrape Specifics
- Uses `wait_until="domcontentloaded"` (NOT `networkidle` — TPT analytics prevent idle state)
- 2-second explicit wait after page load for JS rendering
- Review detection uses JSON-LD `reviewCount`/`ratingCount` (not CSS selectors — those produce false positives)
- Date extraction: ISO dates from JSON-LD `datePublished`, fallback to rendered "Month D, YYYY"
- Products with 0 ratings are bulk-marked via SQL, never visited with a browser

### Patterns to Follow
- All DB writes use `aiosqlite` with `async with` context managers
- Concurrency controlled via `asyncio.Semaphore`
- Progress logging: `Progress: X/Y (Z%) | Found: N | ... | Elapsed: Xm Ys | ETA: Xm Ys | Rate: N/s`
- Errors are logged and counted, never crash the batch — `asyncio.gather(*tasks, return_exceptions=True)`
- Type hints throughout; `Optional`, `Dict`, `List`, `Tuple` from `typing`

## Conventions

- Always run Python code from a script file, never via inline `python -c` commands
- Python 3.13+, no type stub issues expected
- Single `logging.basicConfig` at module level, `logging.info/warning/error` everywhere
- CLI via `argparse` with subparsers: `config create/list`, `search`, `scrape`, `deepscrape`, `download`, `stats`
- Config JSON files use arrays for all filter values (Cartesian product expansion)
- PowerShell as the terminal shell (Windows environment)
- Tests use `pytest` with `asyncio_mode = auto`

## When Modifying Code

- Keep everything in `tpt_scraper.py` unless there's a clear reason to split
- Maintain resume support — new workflows should skip already-completed items
- Use sentinel values in the DB rather than separate tracking tables
- Always handle Playwright context cleanup (`await context.close()`) even on error
- Rate limit awareness: include delays between batches (`asyncio.sleep`)
- Never use `wait_until="networkidle"` for TPT pages
