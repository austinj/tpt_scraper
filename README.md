# TPT Scraper

A workflow-based scraper for Teachers Pay Teachers with support for multiple named configurations.

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

## Quick Start

```powershell
# 1. Create a configuration
python tpt_scraper.py config create my-search --template config.json --description "My search"

# 2. Edit config as needed
notepad configs/my-search.json

# 3. Run the three workflows
python tpt_scraper.py search my-search    # Find product URLs
python tpt_scraper.py scrape my-search    # Extract metadata
python tpt_scraper.py download my-search  # Download free files

# 4. View stats
python tpt_scraper.py stats my-search
```

## Configuration

Configs are JSON files in `configs/`. Example:

```json
{
  "resource_type": ["teacher-tools"],
  "grade_level": ["elementary"],
  "subject": ["social-emotional/classroom-management"],
  "price_options": ["free"],
  "sorting_methods": ["Relevance", "Rating"],
  "total_pages": 20
}
```

Commands:
- `python tpt_scraper.py config list` - List all configs
- `python tpt_scraper.py config create <name> --template config.json` - Create new config

## Download Options

```powershell
# Download all free products
python tpt_scraper.py download my-search

# Download only queued products (manual curation)
python tpt_scraper.py download my-search --use-queue

# Filter by type
python tpt_scraper.py download my-search --filter resource_type=teacher-tools
```

### Manual Queue

Add products to download queue via SQL:

```sql
INSERT INTO download_queue (product_url, priority, notes) 
VALUES ('https://...', 1, 'Good resource');
```

Then run with `--use-queue` flag.

## Database

Each config gets its own SQLite database (`scrape_cache_<config>.db`) with tables:
- `search_results` - Found product URLs
- `product_metadata` - Scraped product details  
- `downloads` - Downloaded files tracking
- `download_queue` - Manual download queue

## Notes

- **Authentication**: Downloads require `tpt_storage.json` with TPT session
- **Resumable**: All workflows can be stopped and resumed
- **Rate Limited**: Adaptive rate limiting built in
