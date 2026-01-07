# Manual Download Queue Workflow

This guide shows how to manually manipulate the database to control which products get downloaded.

## Two Download Modes

### Mode 1: Automatic (Default)
Downloads all free products matching your filters:
```bash
python tpt_scraper.py download coding
python tpt_scraper.py download coding --filter resource_type=teacher-tools
```

### Mode 2: Manual Queue (New)
Downloads only products you've manually added to the `download_queue` table:
```bash
python tpt_scraper.py download coding --use-queue
```

## Manual Queue Workflow

### Step 1: Run Search and Scrape
```bash
python tpt_scraper.py search coding
python tpt_scraper.py scrape coding
```

### Step 2: Query and Analyze Your Data
Open the database and explore what you have:
```bash
sqlite3 scrape_cache_coding.db
```

```sql
-- See all free classroom management products
SELECT 
    m.url,
    m.title,
    m.product_price,
    m.rating_value,
    m.preview_keywords
FROM product_metadata m
JOIN search_results s ON m.url = s.url
WHERE s.price_option = 'free'
LIMIT 10;

-- Count by grade level
SELECT 
    s.grade_level,
    COUNT(*) as count
FROM search_results s
JOIN product_metadata m ON s.url = m.url
WHERE s.price_option = 'free'
GROUP BY s.grade_level
ORDER BY count DESC;

-- Find highly rated products
SELECT 
    m.url,
    m.title,
    m.rating_value,
    m.number_of_ratings
FROM product_metadata m
JOIN search_results s ON m.url = s.url
WHERE s.price_option = 'free'
AND CAST(m.rating_value AS REAL) >= 4.5
AND CAST(m.number_of_ratings AS INTEGER) >= 100
ORDER BY CAST(m.rating_value AS REAL) DESC;
```

### Step 3: Populate Download Queue
Add specific products you want to download:

```sql
-- Add a single product
INSERT INTO download_queue (product_url, priority, notes)
VALUES ('https://www.teacherspayteachers.com/Product/...', 10, 'Highly rated kindergarten resource');

-- Add all kindergarten products
INSERT INTO download_queue (product_url, priority, notes)
SELECT 
    s.url,
    5,
    'Kindergarten - ' || m.title
FROM search_results s
JOIN product_metadata m ON s.url = m.url
WHERE s.grade_level = 'elementary/kindergarten'
AND s.price_option = 'free';

-- Add only 4+ star rated products
INSERT INTO download_queue (product_url, priority, notes)
SELECT 
    s.url,
    CAST(m.rating_value AS REAL) * 2,  -- Use rating as priority
    'Rating: ' || m.rating_value || ' (' || m.number_of_ratings || ' ratings)'
FROM search_results s
JOIN product_metadata m ON s.url = m.url
WHERE s.price_option = 'free'
AND CAST(m.rating_value AS REAL) >= 4.0
AND m.rating_value IS NOT NULL;

-- Add products with specific keywords
INSERT INTO download_queue (product_url, priority, notes)
SELECT 
    s.url,
    8,
    m.preview_keywords
FROM search_results s
JOIN product_metadata m ON s.url = m.url
WHERE s.price_option = 'free'
AND (
    m.preview_keywords LIKE '%behavior%'
    OR m.preview_keywords LIKE '%rewards%'
    OR m.title LIKE '%classroom management%'
);
```

### Step 4: Review Your Queue
```sql
-- See what's queued for download
SELECT 
    q.priority,
    q.product_url,
    q.notes,
    m.title,
    m.rating_value
FROM download_queue q
JOIN product_metadata m ON q.product_url = m.url
ORDER BY q.priority DESC, q.added_at ASC
LIMIT 20;

-- Count items in queue
SELECT COUNT(*) as total_queued FROM download_queue;

-- Remove items from queue if needed
DELETE FROM download_queue WHERE priority < 3;
DELETE FROM download_queue WHERE product_url = 'https://...specific-url...';
```

### Step 5: Download from Queue
```bash
python tpt_scraper.py download coding --use-queue
```

This will download products in order of:
1. Priority (highest first)
2. Date added (oldest first)

## Example: Complex Filtering Workflow

```sql
-- Find products but exclude certain keywords
INSERT INTO download_queue (product_url, priority, notes)
SELECT 
    s.url,
    7,
    'Filtered selection - ' || m.title
FROM search_results s
JOIN product_metadata m ON s.url = m.url
WHERE s.price_option = 'free'
AND s.grade_level IN ('elementary/1st-grade', 'elementary/2nd-grade', 'elementary/3rd-grade')
AND m.title IS NOT NULL
AND m.title NOT LIKE '%digital%'  -- Exclude digital-only
AND m.title NOT LIKE '%Google%'   -- Exclude Google-specific
AND (
    m.rating_value IS NULL 
    OR CAST(m.rating_value AS REAL) >= 4.0
);
```

## Useful Queries for Analysis

```sql
-- Products by format (from preview_keywords)
SELECT 
    CASE 
        WHEN preview_keywords LIKE '%PDF%' THEN 'PDF'
        WHEN preview_keywords LIKE '%PowerPoint%' THEN 'PowerPoint'
        WHEN preview_keywords LIKE '%Google%' THEN 'Google'
        ELSE 'Other'
    END as format_type,
    COUNT(*) as count
FROM product_metadata m
JOIN search_results s ON m.url = s.url
WHERE s.price_option = 'free'
GROUP BY format_type;

-- Products by price (should all be free, but check)
SELECT 
    product_price,
    COUNT(*) as count
FROM product_metadata m
JOIN search_results s ON m.url = s.url
WHERE s.price_option = 'free'
GROUP BY product_price;

-- Check what's already downloaded
SELECT 
    d.product_url,
    d.file_path,
    d.file_size / 1024 / 1024 as size_mb,
    d.downloaded_at
FROM downloads d
ORDER BY d.downloaded_at DESC
LIMIT 20;
```

## Managing the Queue

```sql
-- Clear the entire queue
DELETE FROM download_queue;

-- Update priorities
UPDATE download_queue 
SET priority = 10 
WHERE notes LIKE '%highly rated%';

-- Remove already downloaded items from queue
DELETE FROM download_queue 
WHERE product_url IN (SELECT product_url FROM downloads);

-- Add note to queue items
UPDATE download_queue 
SET notes = notes || ' [PRIORITY]'
WHERE priority >= 8;
```

## Tips

1. **Start small**: Queue just 10-20 products first to test
2. **Use priorities**: 10 = must have, 5 = nice to have, 1 = optional
3. **Check file sizes**: After downloading a batch, check the total size
4. **Track progress**: The `downloads` table shows what's been completed
5. **Clean queue**: Remove downloaded items periodically to keep queue clean

## Resumable

Both download modes are resumable:
- Already downloaded products are skipped
- Queue items that were downloaded are skipped
- Safe to interrupt and restart
