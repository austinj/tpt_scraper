# TPT Scraper

A workflow-based scraper for Teachers Pay Teachers with support for multiple named configurations. Includes search, metadata scraping, deep scraping (first review dates via the evaluations GraphQL API), and bulk downloading.

## Setup

```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1    # Windows PowerShell
# or: source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
playwright install chromium
```

### Authentication

The download workflow requires a valid TPT session stored in `tpt_storage.json`. Generate it with:

```powershell
python create_session.py
```

## Quick Start

```powershell
# 1. Create a configuration
python tpt_scraper.py config create my-search --template config.json --description "My search"

# 2. Edit config as needed
notepad configs/my-search.json

# 3. Run the four workflows in order
python tpt_scraper.py search my-search       # Find product URLs
python tpt_scraper.py scrape my-search       # Extract metadata
python tpt_scraper.py deepscrape my-search   # Get first review dates (GraphQL API)
python tpt_scraper.py download my-search     # Download free files

# 4. View stats
python tpt_scraper.py stats my-search
```

## Workflows

### 1. Search

Discovers product URLs by iterating over all combinations of config filters (resource type, grade level, subject, price, sort order) across paginated results.

Before the full run a pre-flight pass checks each base URL, then verifies every configured sort order against the server: TPT silently falls back to Relevance for unknown `order=` values, so any sort whose page-1 results match Relevance is dropped (and logged) instead of re-fetching the same pages. Currently valid: `Relevance`, `Rating`, `Price-Asc`, `Most-Recent`.

Two tables are written. `search_results` keeps one row per unique URL (the first combination that found it) and is what the later stages read. `search_hits` keeps **every** appearance - one row per product x grade x subject x price x sort x page, with its rank on the page - so you can measure how many searches surface the same product. The `product_search_summary` view aggregates this per product (`n_hits`, distinct grades/subjects/prices/sorts, best page/position).

```sql
-- products found by the most search combinations
SELECT product_id, n_hits, n_grade_levels, n_subjects, sort_orders, best_page
FROM product_search_summary ORDER BY n_hits DESC LIMIT 20;
```

```powershell
python tpt_scraper.py search my-search
python tpt_scraper.py search my-search --auto-scrape   # Start metadata scrape automatically after search
```

### 2. Scrape (Metadata)

Fetches metadata for each discovered product URL via HTTP (no browser needed). Extracts title, description, rating, price, author info from the HTML, plus everything in the page's embedded Apollo GraphQL cache (`"apolloState"`): the product's posting date, file type/size, license prices, taxonomy tags, bundle membership, exact seller follower count and catalog size, and the review facet counts (star distribution, grades used with, SPED use, yes/no satisfaction signals).

```powershell
python tpt_scraper.py scrape my-search
python tpt_scraper.py scrape my-search --rescrape   # Re-scrape ALL products, updating existing records
python tpt_scraper.py scrape my-search --backfill   # Re-scrape only existing products missing post_date (resumable)
python tpt_scraper.py scrape my-search --no-cache   # Bypass HTTP cache for fresh fetches
```

`--backfill` is for databases scraped before the Apollo-state columns existed. It re-fetches each page and fills the new columns, leaving `preview_keywords` untouched (the grade label it was built from is client-rendered and not reliably present in a plain fetch; use `taxonomy_tags` instead).

### 3. Deep Scrape (First Review Date)

Queries the reviews widget's GraphQL endpoint (`/gateway/graphql`, `filterEvaluationsByResource`) directly, with no browser or login. Cursors are base64-encoded integer offsets, so the scraper sorts by most recent and jumps straight to `number_of_ratings - 100` to read the oldest reviews in one request (two if new reviews arrived since the metadata scrape). Products with 0 ratings are bulk-marked `no_reviews` and skipped.

Writes `first_review_date` (`YYYY-MM-DD`). Failed lookups are left NULL and retried on the next run.

> The legacy `oldest_review_date` column (Playwright-era) is **not** a first-review date: it read dates from the page's JSON-LD, which only lists the ~10 most recent reviews. Use `first_review_date` or `post_date` instead.

```powershell
python tpt_scraper.py deepscrape my-search
python tpt_scraper.py deepscrape my-search --limit 50        # Test with a small batch
python tpt_scraper.py deepscrape my-search --concurrent 8    # Concurrent API requests (default: 4)
python tpt_scraper.py deepscrape my-search --force           # Recompute for ALL products
```

### 4. Download

Downloads free product files using Playwright for authenticated access.

```powershell
python tpt_scraper.py download my-search
python tpt_scraper.py download my-search --use-queue    # Download only queued products
python tpt_scraper.py download my-search --filter resource_type=teacher-tools
python tpt_scraper.py download my-search --extract      # Auto-extract zip files
python tpt_scraper.py download my-search --concurrent 5 # Concurrent downloads (default: 5)
```

### 5. Stats

```powershell
python tpt_scraper.py stats my-search
```

## Configuration

Configs are JSON files in `configs/`, registered in `configs/registry.json`. Each config maps to its own SQLite database.

```json
{
  "resource_type": ["teacher-tools"],
  "grade_level": ["elementary", "middle-school"],
  "subject": ["social-emotional/classroom-management"],
  "price_options": ["free", "paid"],
  "sorting_methods": ["Relevance", "Rating", "Price-Asc", "Most-Recent"],
  "total_pages": 20
}
```

All filter arrays are combined as a Cartesian product — every combination is searched. More values = exponentially more search batches.

Commands:
- `python tpt_scraper.py config list` — List all configs with their DB file and description
- `python tpt_scraper.py config create <name> --template config.json` — Create new config from template

## Database Schema

Each config gets its own SQLite database (`scrape_cache_<config>.db`) with WAL mode enabled.

| Table | Purpose |
|-------|---------|
| `search_results` | Unique product URLs with `product_id` and the first search combination that found them |
| `search_hits` | Every (product, search combination, page, rank) appearance - for overlap analysis |
| `product_search_summary` | View: per-product overlap summary over `search_hits` |
| `product_metadata` | Scraped product details (title, rating, price, author, posting date, first review date, etc.) |
| `downloads` | Downloaded file tracking |
| `download_queue` | Manual curation queue for selective downloads |

### Metadata Fields

| Column | Type | Source |
|--------|------|--------|
| `title` | TEXT | Scrape |
| `short_description` | TEXT | Scrape |
| `long_description` | TEXT | Scrape |
| `rating_value` | TEXT | Scrape |
| `number_of_ratings` | TEXT | Scrape |
| `product_price` | TEXT | Scrape |
| `preview_keywords` | TEXT | Scrape |
| `author_name` | TEXT | Scrape |
| `author_store_url` | TEXT | Scrape |
| `author_follower_count` | INTEGER | Scrape (rounded, e.g. "51K" -> 51000) |
| `post_date` | TEXT | Scrape (Apollo) - when the product was posted, `YYYY-MM-DD HH:MM:SS` |
| `item_type` | TEXT | Scrape (Apollo) - `DIGITAL_PRODUCT`, `BUNDLE`, `ONLINE_RESOURCE`, ... |
| `file_kind` | TEXT | Scrape (Apollo) - `PDF`, `ZIP`, `PPTX`, ... |
| `file_size_bytes` | INTEGER | Scrape (Apollo) |
| `page_count` | TEXT | Scrape (Apollo) - e.g. `11`, `35+ Pages` |
| `price_full` / `price_discount` / `price_sale` / `price_additional_license` | TEXT | Scrape (Apollo) - `$6.00` strings |
| `is_free` | INTEGER | Scrape (Apollo) |
| `taxonomy_tags` | TEXT | Scrape (Apollo) - comma-separated slugs: grades, subjects, resource types, formats |
| `common_core_standards` | TEXT | Scrape (Apollo) - JSON list |
| `answer_key` / `teaching_duration` | TEXT | Scrape (Apollo) |
| `bundle_count` | INTEGER | Scrape (Apollo) - number of bundles this product is included in |
| `image_count` / `has_preview_video` / `has_main_video` | INTEGER | Scrape (Apollo) |
| `store_id` / `store_follower_count` / `store_total_items` | TEXT / INTEGER | Scrape (Apollo) - exact follower count and seller catalog size |
| `v1_evaluations_count` | INTEGER | Scrape (Apollo) - reviews on the newer structured review form |
| `rating_distribution` | TEXT | Scrape (Apollo) - JSON `{"5": 10596, "4": 874, ...}` |
| `evaluation_facets` | TEXT | Scrape (Apollo) - JSON of all review facets (star, grades used with, SPED, yes/no signals) |
| `first_review_date` | TEXT | Deep scrape - date of the earliest review, or `no_reviews` |
| `oldest_review_date` | TEXT | Legacy deep scrape (Playwright) - **deprecated**, see section 3 |

### Manual Download Queue

```sql
INSERT INTO download_queue (product_url, priority, notes) 
VALUES ('https://...', 1, 'Good resource');
```

Then run with `--use-queue` flag.

## Resumability

All workflows are fully resumable — stop with Ctrl+C and re-run the same command to pick up where you left off:

- **Search**: Skips already-fetched page combinations
- **Scrape**: Skips products that already have metadata (`--backfill`: skips products that already have `post_date`)
- **Deep scrape**: Skips products with any `first_review_date` value (a date or `no_reviews`); failed lookups stay NULL and are retried
- **Download**: Skips already-downloaded files

## Notes

- **Authentication**: Only downloads require `tpt_storage.json` (see Setup)
- **Rate Limited**: Adaptive rate limiting and concurrency controls built in
- **Progress Tracking**: All workflows display real-time progress with ETA, rate, and counts
- **SQLite WAL Mode**: Safe for concurrent reads while a workflow is writing
