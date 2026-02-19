"""
Find all oldest_review_date values before 2006 (impossible — TPT launched 2006),
reset them to NULL, then re-scrape those products.
"""
import sqlite3
from datetime import datetime

DB = "scrape_cache_sel_simple.db"
CONFIG = "sel_simple"

conn = sqlite3.connect(DB)
rows = conn.execute("""
    SELECT rowid, oldest_review_date
    FROM product_metadata
    WHERE oldest_review_date NOT IN ('no_reviews', 'parse_failed')
      AND oldest_review_date IS NOT NULL
""").fetchall()

bad = []
for rowid, d in rows:
    try:
        parsed = datetime.strptime(d, "%Y-%m-%d")
    except ValueError:
        try:
            parsed = datetime.strptime(d, "%B %d, %Y")
        except ValueError:
            continue
    if parsed.year < 2006:
        bad.append((rowid, d))

print(f"Found {len(bad)} rows with dates before 2006:")
for rowid, d in bad:
    print(f"  rowid={rowid}  date={d}")

if bad:
    ids = [r[0] for r in bad]
    placeholders = ",".join("?" * len(ids))
    conn.execute(
        f"UPDATE product_metadata SET oldest_review_date = NULL WHERE rowid IN ({placeholders})",
        ids,
    )
    print(f"\nReset {conn.total_changes} rows to NULL")
    conn.commit()
else:
    print("\nNo bad dates found — nothing to reset.")

conn.close()

if bad:
    import subprocess, sys
    print(f"\nRunning: python tpt_scraper.py deepscrape {CONFIG} --limit {len(bad)}")
    subprocess.run([sys.executable, "tpt_scraper.py", "deepscrape", CONFIG, "--limit", str(len(bad))])
