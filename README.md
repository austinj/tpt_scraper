# TPT Scraper

A workflow-based scraper for Teachers Pay Teachers with support for multiple named configurations. Includes search, metadata scraping, deep scraping (oldest review dates via Playwright), and bulk downloading.

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

Several workflows (deep scrape, download) require a valid TPT session stored in `tpt_storage.json`. Generate it with:

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
python tpt_scraper.py deepscrape my-search   # Get oldest review dates (Playwright)
python tpt_scraper.py download my-search     # Download free files

# 4. View stats
python tpt_scraper.py stats my-search
```

## Workflows

### 1. Search

Discovers product URLs by iterating over all combinations of config filters (resource type, grade level, subject, price, sort order) across paginated results.

```powershell
python tpt_scraper.py search my-search
python tpt_scraper.py search my-search --auto-scrape   # Start metadata scrape automatically after search
```

### 2. Scrape (Metadata)

Fetches metadata for each discovered product URL via HTTP (no browser needed). Extracts title, description, rating, price, author info, and more.

```powershell
python tpt_scraper.py scrape my-search
python tpt_scraper.py scrape my-search --rescrape   # Re-scrape ALL products, updating existing records
python tpt_scraper.py scrape my-search --no-cache   # Bypass HTTP cache for fresh fetches
```

### 3. Deep Scrape (Oldest Review Date)

Uses Playwright (headless Chromium) to visit each product page and extract the oldest review date. Products with 0 ratings are bulk-marked and skipped — only products with ratings are visited.

- Parses JSON-LD structured data (`reviewCount`, `datePublished`) for reliable detection
- Falls back to rendered "Month D, YYYY" date text in review sections
- Clicks "Show more reviews" to paginate through all reviews
- Writes sentinel values (`no_reviews`, `parse_failed`) for resume support

```powershell
python tpt_scraper.py deepscrape my-search
python tpt_scraper.py deepscrape my-search --limit 50        # Test with a small batch
python tpt_scraper.py deepscrape my-search --concurrent 5    # More browser pages (default: 3)
python tpt_scraper.py deepscrape my-search --session-file auth.json
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
  "sorting_methods": ["Relevance", "Rating"],
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
| `search_results` | Discovered product URLs with search parameters |
| `product_metadata` | Scraped product details (title, rating, price, author, oldest review date, etc.) |
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
| `author_follower_count` | INTEGER | Scrape |
| `oldest_review_date` | TEXT | Deep scrape |

### Manual Download Queue

```sql
INSERT INTO download_queue (product_url, priority, notes) 
VALUES ('https://...', 1, 'Good resource');
```

Then run with `--use-queue` flag.

## Resumability

All workflows are fully resumable — stop with Ctrl+C and re-run the same command to pick up where you left off:

- **Search**: Skips already-fetched page combinations
- **Scrape**: Skips products that already have metadata
- **Deep scrape**: Skips products with any `oldest_review_date` value (including sentinels `no_reviews` / `parse_failed`)
- **Download**: Skips already-downloaded files

## Notes

- **Authentication**: Deep scrape and downloads require `tpt_storage.json` (see Setup)
- **Rate Limited**: Adaptive rate limiting and concurrency controls built in
- **Progress Tracking**: All workflows display real-time progress with ETA, rate, and counts
- **SQLite WAL Mode**: Safe for concurrent reads while a workflow is writing
