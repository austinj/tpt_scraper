# TPT Scraper - Refactored Version

A clean, workflow-based scraper for Teachers Pay Teachers with support for multiple named configurations.

## ✨ Key Improvements

### Clear Workflow Separation
- **Search**: Find product URLs matching your criteria
- **Scrape**: Extract metadata for discovered products  
- **Download**: Download free product files

### Named Configurations
- Create multiple configurations for different searches
- Each configuration has its own isolated database
- Easy to manage and switch between different search strategies

### Better Data Organization
- Separate databases per configuration
- Clear separation of search results, metadata, and downloads
- Easy to export and analyze specific searches

## 🚀 Quick Start

### 1. Create a Configuration

```bash
# Create from existing config.json
python tpt_scraper_refactored.py config create my-search --template config.json --description "Social emotional learning resources"

# Or create a minimal config (you can edit configs/my-search.json later)
python tpt_scraper_refactored.py config create test-search --description "Test configuration"
```

### 2. Run the Three Workflows

```bash
# Step 1: Search for product URLs
python tpt_scraper_refactored.py search my-search

# Step 2: Scrape metadata for found products
python tpt_scraper_refactored.py scrape my-search

# Step 3: Download free products
python tpt_scraper_refactored.py download my-search
```

### 3. View Statistics

```bash
python tpt_scraper_refactored.py stats my-search
```

## 📋 Configuration Management

### List All Configurations

```bash
python tpt_scraper_refactored.py config list
```

### Edit a Configuration

Configurations are stored as JSON files in the `configs/` directory:

```bash
# Edit the configuration file directly
notepad configs/my-search.json
```

### Configuration File Format

```json
{
  "resource_type": ["teacher-tools"],
  "grade_level": ["elementary", "middle-school"],
  "subject": ["social-emotional/classroom-management"],
  "format": ["pdf"],
  "price_options": ["free"],
  "supports": [""],
  "sorting_methods": ["Relevance", "Rating"],
  "total_pages": 20,
  "concurrent_requests": 25
}
```

## 🎯 Workflows

### 1. Search Workflow

Discovers product URLs based on your configuration parameters.

```bash
python tpt_scraper_refactored.py search <config-name>
```

**What it does:**
- Generates all combinations of your search parameters
- Fetches search result pages from TPT
- Extracts product URLs
- Stores URLs in `scrape_cache_<config-name>.db`
- Tracks which combinations have been searched (resumable)

**Output:** `search_results` table with all discovered URLs

### 2. Scrape Metadata Workflow

Extracts detailed information from product pages.

```bash
python tpt_scraper_refactored.py scrape <config-name>
```

**What it does:**
- Gets all URLs from search results that haven't been scraped
- Scrapes each product page for metadata
- Stores title, descriptions, ratings, price, keywords

**Output:** `product_metadata` table with product details

### 3. Download Free Files Workflow

Downloads actual product files for free resources.

```bash
# Download all free products
python tpt_scraper_refactored.py download <config-name>

# Download with filters
python tpt_scraper_refactored.py download <config-name> --filter resource_type=teacher-tools
python tpt_scraper_refactored.py download <config-name> --filter subject=classroom-management
```

**What it does:**
- Finds all free products in your search results
- Downloads files using Playwright automation
- Saves to `downloads_<config-name>/` directory
- Tracks downloads to avoid duplicates

**Output:** Downloaded files in `downloads_<config-name>/`

## 📊 Example: Classroom Management Free Resources

```bash
# 1. Create a targeted configuration
cat > classroom-mgmt-config.json << EOF
{
  "resource_type": ["teacher-tools"],
  "grade_level": [
    "elementary/kindergarten",
    "elementary/1st-grade",
    "elementary/2nd-grade",
    "elementary/3rd-grade",
    "elementary/4th-grade",
    "elementary/5th-grade"
  ],
  "subject": ["social-emotional/classroom-management"],
  "format": [""],
  "price_options": ["free"],
  "supports": [""],
  "sorting_methods": ["Relevance", "Rating"],
  "total_pages": 30,
  "concurrent_requests": 25
}
EOF

python tpt_scraper_refactored.py config create classroom-mgmt \
  --template classroom-mgmt-config.json \
  --description "Free classroom management resources for elementary"

# 2. Run the workflows
python tpt_scraper_refactored.py search classroom-mgmt
python tpt_scraper_refactored.py scrape classroom-mgmt
python tpt_scraper_refactored.py download classroom-mgmt

# 3. Check results
python tpt_scraper_refactored.py stats classroom-mgmt
```

## 🗂️ File Organization

```
tpt_scraper/
├── config_manager.py              # Configuration management
├── tpt_scraper_refactored.py      # Main refactored script
├── configs/                        # Named configurations
│   ├── registry.json              # Config registry
│   ├── classroom-mgmt.json        # Example config
│   └── my-search.json             # Another config
├── scrape_cache_classroom-mgmt.db # Database for classroom-mgmt config
├── scrape_cache_my-search.db      # Database for my-search config
├── downloads_classroom-mgmt/      # Downloads for classroom-mgmt
└── downloads_my-search/           # Downloads for my-search
```

## 🔧 Advanced Usage

### Resume Interrupted Searches

The search workflow automatically tracks what's been searched. If interrupted:

```bash
# Just run again - it will continue where it left off
python tpt_scraper_refactored.py search my-search
```

### Incremental Scraping

Only scrapes URLs that don't have metadata yet:

```bash
# Add more search results, then scrape only new ones
python tpt_scraper_refactored.py scrape my-search
```

### Filtered Downloads

Download only specific types of free products:

```bash
# Only download teacher-tools
python tpt_scraper_refactored.py download my-search --filter resource_type=teacher-tools

# Only download kindergarten resources  
python tpt_scraper_refactored.py download my-search --filter grade_level=elementary/kindergarten
```

### Custom Concurrent Requests

```bash
# Download with more concurrent downloads
python tpt_scraper_refactored.py download my-search --concurrent 10
```

## 🆚 Comparison with Original Script

### Original Script
- Single monolithic workflow
- One database for everything
- Mix of search/scrape/download logic
- Hard to manage different searches

### Refactored Script
- ✅ Three clear, independent workflows
- ✅ Multiple named configurations
- ✅ Isolated databases per config
- ✅ Easy to run specific stages
- ✅ Better for targeted searches

## 🔄 Migration from Original Script

If you have data in the original `scrape_cache.db`:

```bash
# 1. Create a config representing your old search
python tpt_scraper_refactored.py config create legacy --template config.json

# 2. Copy your old database
cp scrape_cache.db scrape_cache_legacy.db

# 3. Use the new workflows going forward
python tpt_scraper_refactored.py stats legacy
```

## 📝 Notes

- **Authentication**: For downloads, you still need `tpt_storage.json` with your TPT login session
- **Rate Limiting**: Adaptive rate limiting is built in to avoid overwhelming TPT
- **Resumable**: All workflows can be stopped and resumed safely
- **Concurrent**: Configurable concurrency for both searching and scraping

## 🎓 Typical Workflow

```bash
# 1. Define what you want to search for
python tpt_scraper_refactored.py config create science-5th \
  --description "5th grade science resources"

# 2. Edit the config to be specific
notepad configs/science-5th.json

# 3. Search for products
python tpt_scraper_refactored.py search science-5th

# 4. Get detailed metadata
python tpt_scraper_refactored.py scrape science-5th

# 5. Download free resources
python tpt_scraper_refactored.py download science-5th

# 6. Analyze your data
python tpt_scraper_refactored.py stats science-5th
# Or query the database directly:
# sqlite3 scrape_cache_science-5th.db "SELECT * FROM product_metadata WHERE product_price = '0.00'"
```
