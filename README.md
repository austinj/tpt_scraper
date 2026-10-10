# TPT Scraper

Collects product data from Teachers Pay Teachers (TPT) for research. Four resumable stages share one SQLite database per named configuration:

| Stage | Command | What it does | Needs |
|---|---|---|---|
| 1 | `search` | Enumerates product URLs through TPT's search API | HTTP only |
| 2 | `scrape` | Fetches each product page: title, description, rating, price, seller, posting date, taxonomy tags, review facets, ... | HTTP only |
| 3 | `deepscrape` | Looks up each rated product's first review date through the reviews API | HTTP only |
| 4 | `download` | Downloads free product files | Playwright + TPT login |

Everything is plain HTTP except downloads. The two API stages call the same `/gateway/graphql` endpoint the site's own front end uses.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # or: source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium           # only needed for `download`
pip install pytest pytest-asyncio     # only needed to run the tests
```

For `download`, store a logged-in TPT session once with `python create_session.py` (writes `tpt_storage.json`).

## Quick start

```powershell
# Create a configuration (copy an existing one as the template, then edit configs/<name>.json)
python tpt_scraper.py config create sel_v2 --template configs\sel_simple.json --description "SEL rescrape"

# Run the stages in order. Each is safe to Ctrl+C and re-run.
python tpt_scraper.py search     sel_v2
python tpt_scraper.py scrape     sel_v2 *>> scrape_sel_v2.log
python tpt_scraper.py deepscrape sel_v2 *>> deepscrape_sel_v2.log
python tpt_scraper.py stats      sel_v2
```

Rough timings for the SEL configuration (~175k products, `concurrent_requests: 8`): search ~15 min, scrape ~10-12 h, deepscrape ~4 h. The scrape and deepscrape log one line per product, so redirecting to a file (`*>>`) is recommended; follow along with `Get-Content scrape_sel_v2.log -Wait -Tail 20`.

## Configuration

Configs live in `configs/<name>.json` and are registered in `configs/registry.json`, which maps each name to its database `scrape_cache_<name>.db`. Create one with `config create`; list them with `config list`.

```json
{
  "resource_type":  [""],
  "grade_level":    ["", "elementary", "elementary/preschool", "middle-school/6th-grade", "not-grade-specific"],
  "subject":        ["social-emotional", "social-emotional/character-education"],
  "format":         [""],
  "price_options":  ["", "free", "under-5", "5-to-10", "above-10", "on-sale"],
  "supports":       [""],
  "sorting_methods": ["Relevance", "Rating", "Price-Asc", "Most-Recent"],

  "concurrent_requests": 8,
  "batch_size": 75,
  "sleep_between_batches": [0.5, 2.0],
  "max_rate_limit_delay": 30.0
}
```

**Facets.** `resource_type`, `grade_level`, `subject`, `format`, `price_options`, `supports` are lists of TPT browse-URL path fragments; `""` means "no filter on this dimension". The search runs every combination (Cartesian product) under every sort order, so the number of queries is the product of the list lengths. Only the last path segment matters to the API (`elementary/preschool` -> `preschool`); the longer forms are accepted for readability.

**Sort orders.** TPT currently offers `Relevance`, `Rating`, `Price-Asc`, `Most-Recent`. Anything else is dropped with a warning (older configs listed `Rating-Count` and `Price-Desc`, which the server silently treated as Relevance).

**Throughput.** `concurrent_requests` caps in-flight requests (8 is sensible; 25 provoked Cloudflare). `batch_size` is how many items are gathered before a DB commit and progress line. `sleep_between_batches` is a random pause range in seconds. `max_rate_limit_delay` caps the adaptive back-off.

`total_pages` in older configs is ignored: the API returns a query's whole result set (up to the cap, below) in one request.

## Stages in detail

### 1. Search

Enumerates products via `searchResources` on `/gateway/graphql` - the call the browse page makes when you click "next page" - rather than scraping browse HTML. One request per (combination x sort order) returns up to **1,000** results; that cap is the same one the HTML browse pages have (42 pages x 24), and it is why the config fans out over grades x subjects x prices x sorts.

Why not the HTML pages: Cloudflare puts a managed challenge in front of browse URLs for deeper facet combinations (a specific grade + subject + price, among others). It is per-URL and permanent for any HTTP client - not a rate limit, and not solvable by a headless browser. The API serves those combinations normally. As a bonus, one request replaces 42 page loads: the SEL search is ~2,700 requests instead of ~115,000.

Pre-flight runs a handful of requests before any searching:
- connectivity + catalogue size (one unfaceted query);
- **every facet slug must narrow the catalogue** - the API silently ignores slugs it doesn't recognise, so a typo would return all 17M products as if filtered. Any unknown slug aborts the run with the offending values listed;
- each sort order must produce a distinct ordering from Relevance; duplicates are dropped.

Tables written:
- `search_results` - one row per unique URL with `product_id` and the first combination that found it. Later stages read this.
- `search_hits` - **every** appearance: one row per product x combination x sort with its `rank_on_page`. This is what lets you measure overlap between searches.
- `search_queries` - one row per completed query: `facets`, `total_count`, `returned`, `capped`. A capped query had more matches than one request can return; its remaining products are only reachable through other combinations. Failed queries are not recorded, so re-running retries them.
- `product_search_summary` (view) - per product: `n_hits`, distinct grades/subjects/prices/sorts, best position, and the lists of combinations it appeared in.

```sql
-- products surfaced by the most search combinations
SELECT product_id, n_hits, n_grade_levels, n_subjects, sort_orders FROM product_search_summary ORDER BY n_hits DESC LIMIT 20;
-- combinations that overflowed the cap, largest first
SELECT facets, sort_order, total_count FROM search_queries WHERE capped = 1 ORDER BY total_count DESC;
```

### 2. Scrape (metadata)

Fetches each product page once, over plain HTTP, and reads two things from it:

- the visible HTML: title, descriptions, rating, rating count, price, seller name/store URL, rounded follower count ("51K"), and `preview_keywords` (category links);
- the page's embedded Apollo GraphQL cache (`"apolloState"`), which carries the structured product record: posting date, item type, file type/size/page count, full license prices, taxonomy tags, bundle membership, media counts, the seller's exact follower count and catalogue size, and the review facet counts (star distribution, grades used with, special-education use, yes/no satisfaction signals).

```powershell
python tpt_scraper.py scrape my-search
python tpt_scraper.py scrape my-search --backfill   # only existing rows missing post_date (for DBs scraped before the Apollo fields existed)
python tpt_scraper.py scrape my-search --rescrape   # re-fetch ALL products, overwriting
python tpt_scraper.py scrape my-search --no-cache   # bypass the 1-hour HTTP cache
```

Each product logs one line: `[2495504] ratings=632 posted=2016-04-10 followers=2068 price=$3.50 | <title>`. `posted=MISSING` means the Apollo state wasn't found - if that appears for many products in a row, TPT has changed its page and the parser needs attention.

`--backfill` leaves `preview_keywords` untouched: the grade label it was partly built from is rendered client-side and rarely present in a plain fetch. Use `taxonomy_tags` for grades/subjects/formats; it is populated for every product.

### 3. Deep scrape (first review date)

Finds the date of each rated product's earliest review via `filterEvaluationsByResource` on `/gateway/graphql`. Review cursors are base64-encoded integer offsets, so the scraper sorts by most recent and jumps straight to `number_of_ratings - 100`, reading the oldest reviews in one request (two if new reviews arrived since the metadata scrape; the server's 10,000 `totalCount` display cap does not limit offsets). Products with 0 ratings are bulk-marked `no_reviews` without a request.

Writes `first_review_date` as `YYYY-MM-DD`. Failed lookups stay NULL and are retried next run.

```powershell
python tpt_scraper.py deepscrape my-search
python tpt_scraper.py deepscrape my-search --limit 200     # smoke test
python tpt_scraper.py deepscrape my-search --concurrent 8  # default 4
python tpt_scraper.py deepscrape my-search --force         # recompute for ALL products
```

> **`oldest_review_date` is deprecated.** The Playwright-era deep scrape read dates from the page's JSON-LD, which lists only the ~10 most recent reviews, so that column is roughly "date of the 10th most recent review" - off by years for popular products. Use `post_date` (product age) or `first_review_date` (time since first rating).

### 4. Download

Downloads free product files using Playwright with the stored session.

```powershell
python tpt_scraper.py download my-search
python tpt_scraper.py download my-search --top 500                      # N most-rated free products
python tpt_scraper.py download my-search --use-queue                    # only rows in download_queue
python tpt_scraper.py download my-search --filter resource_type=teacher-tools
python tpt_scraper.py download my-search --concurrent 3                 # default 3
python tpt_scraper.py download my-search --session-file tpt_storage.json
```

Queue specific products with `INSERT INTO download_queue (product_url, priority, notes) VALUES ('https://...', 1, 'why');`.

### 5. Stats

`python tpt_scraper.py stats my-search` prints URL counts, search-overlap buckets, how many products have `post_date` / `first_review_date` (and how many rated products are still pending), and download totals.

## Database schema

One SQLite file per config, WAL mode. Columns are added by migration on every run, so older databases pick up new fields automatically (`scrape --backfill` fills them).

| Table | Purpose |
|---|---|
| `search_results` | Unique product URLs, `product_id`, first combination that found them |
| `search_hits` | Every (product x combination x sort) appearance with rank |
| `search_queries` | Completed search queries with `total_count` / `returned` / `capped` |
| `product_search_summary` | View over `search_hits`: per-product overlap summary |
| `product_metadata` | One row per scraped product (fields below) |
| `downloads`, `download_queue` | Download tracking and manual queue |
| `searched_combinations`, `search_combo_totals` | Legacy (HTML-era search); unused, may exist in older DBs |

### `product_metadata` fields

| Column | Source | Notes |
|---|---|---|
| `url`, `product_id` | scrape | `product_id` is the numeric id from the URL |
| `title`, `short_description`, `long_description` | scrape (HTML) | |
| `rating_value`, `number_of_ratings` | scrape (HTML meta) | TEXT; cast to numeric in analysis |
| `product_price` | scrape (HTML meta) | plain number as text |
| `preview_keywords` | scrape (HTML) | comma-separated category labels; see `taxonomy_tags` |
| `author_name`, `author_store_url` | scrape (HTML) | |
| `author_follower_count` | scrape (HTML) | rounded ("51K" -> 51000); prefer `store_follower_count` |
| `post_date` | scrape (Apollo) | when the product was posted, `YYYY-MM-DD HH:MM:SS` |
| `item_type` | scrape (Apollo) | `DIGITAL_PRODUCT`, `BUNDLE`, `ONLINE_RESOURCE`, ... |
| `file_kind`, `file_size_bytes`, `page_count` | scrape (Apollo) | `page_count` is text, e.g. `11` or `35+ Pages` |
| `price_full`, `price_discount`, `price_sale`, `price_additional_license` | scrape (Apollo) | `$6.00` strings |
| `is_free` | scrape (Apollo) | 0/1 |
| `taxonomy_tags` | scrape (Apollo) | comma-separated slugs: grades, subjects, resource types, formats |
| `common_core_standards` | scrape (Apollo) | JSON list, NULL if none |
| `answer_key`, `teaching_duration` | scrape (Apollo) | |
| `bundle_count` | scrape (Apollo) | number of bundles that include this product |
| `image_count`, `has_preview_video`, `has_main_video` | scrape (Apollo) | |
| `store_id`, `store_follower_count`, `store_total_items` | scrape (Apollo) | exact follower count; seller catalogue size |
| `v1_evaluations_count` | scrape (Apollo) | reviews on the newer structured review form |
| `rating_distribution` | scrape (Apollo) | JSON `{"5": 10596, "4": 874, ...}` |
| `evaluation_facets` | scrape (Apollo) | JSON of all review facets (star, grades used with, SPED, yes/no signals) |
| `first_review_date` | deepscrape | `YYYY-MM-DD`, or `no_reviews` |
| `oldest_review_date` | legacy | **deprecated** - see stage 3 |
| `scraped_at` | scrape | timestamp of the metadata fetch (the "as of" date for follower counts and ratings) |

## Blocks and rate limiting

All HTTP (page fetches and GraphQL posts alike) goes through one `AdaptiveRateLimiter` per run. It serialises dispatch so concurrent workers never fire at once, spaces requests by an adaptive, jittered delay (floor 0.25 s, about 4 req/s), and backs off when errors rise.

A **403, 429, or Cloudflare "Just a moment..." page** is treated as a block rather than a failure:

- every worker pauses together until a shared cool-down expires - 1 min for the first block, doubling per consecutive block, capped at 30 min;
- requests resume at half the previous speed and ease back up only if the site stays quiet;
- the blocked request is retried (up to 6 cool-downs) and is not marked done until it succeeds, so a re-run picks up exactly where it stopped;
- the escalation resets after 300 consecutive successes.

Reading the log: repeated blocks on *different* URLs are an IP-level challenge - let the cool-down run; in testing these cleared within minutes, and lowering `concurrent_requests` is the right response. Repeated blocks on the *same* URL are a URL-level rule (as with the browse HTML pages described under Search) and waiting will not help; that is what moved the search stage onto the API.

## Resumability

Every stage can be stopped with Ctrl+C and re-run with the same command:

- **search** skips queries recorded in `search_queries`; failed/blocked ones were not recorded and run again. The final log warns if any did not complete.
- **scrape** skips products already in `product_metadata` (`--backfill`: skips rows that already have `post_date`).
- **deepscrape** skips products with any `first_review_date` (a date or `no_reviews`); NULLs are retried.
- **download** skips products whose folder already contains files.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest test_tpt_scraper.py -q
```

Unit tests cover the Apollo-state parser, facet/sort validation, the search workflow end-to-end against a fake API (including resume), review-cursor arithmetic, and the block/cool-down logic. No network access is needed. The other `test_*.py` files are older ad-hoc scripts that hit the live site.

## Other files

- `config_manager.py` - config registry.
- `create_session.py` - opens a browser to log in and saves `tpt_storage.json` for downloads.
- `run_pipeline.ps1` - example end-to-end run (search -> scrape -> download top 500) for the `coding` config.
- `check_db.py`, `check_deepscrape.py`, `fix_bad_dates.py` - one-off inspection/repair scripts from the Playwright-era deep scrape.
- `tptscrape_legacy.py` - the original single-file scraper; kept for reference, not used.
