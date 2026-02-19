"""
Quick test: reset 100 products and re-deepscrape them, then check if the
rendered-date fallback ever found a date that JSON-LD didn't.
"""
import sqlite3
import subprocess
import re
import sys

DB = "scrape_cache_sel_simple.db"
CONFIG = "sel_simple"
LIMIT = 100

# 1. Reset 100 products
conn = sqlite3.connect(DB)
conn.execute(f"""
    UPDATE product_metadata
    SET oldest_review_date = NULL
    WHERE rowid IN (
        SELECT rowid FROM product_metadata
        WHERE oldest_review_date NOT IN ('no_reviews', 'parse_failed')
        AND oldest_review_date IS NOT NULL
        LIMIT {LIMIT}
    )
""")
reset = conn.total_changes
conn.commit()
conn.close()
print(f"Reset {reset} rows")

if reset == 0:
    print("Nothing to reset — all rows are already NULL, no_reviews, or parse_failed.")
    sys.exit(0)

# 2. Run deepscrape capturing output
print(f"\nRunning deepscrape {CONFIG} --limit {LIMIT} ...\n")
result = subprocess.run(
    [sys.executable, "tpt_scraper.py", "deepscrape", CONFIG, "--limit", str(LIMIT)],
    capture_output=True, text=True
)

output = result.stdout + "\n" + result.stderr
print(output)

# 3. Analyze results
fallback_only = re.findall(r"FALLBACK-ONLY.*", output)
fallback_older = re.findall(r"FALLBACK-OLDER.*", output)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Products scraped:    {LIMIT}")
print(f"FALLBACK-ONLY hits:  {len(fallback_only)}  (JSON-LD had nothing, fallback found a date)")
print(f"FALLBACK-OLDER hits: {len(fallback_older)}  (fallback found older date than JSON-LD)")
print(f"Total fallback used: {len(fallback_only) + len(fallback_older)}")

if fallback_only:
    print("\nFALLBACK-ONLY lines:")
    for line in fallback_only:
        print(f"  {line}")

if fallback_older:
    print("\nFALLBACK-OLDER lines:")
    for line in fallback_older:
        print(f"  {line}")

if not fallback_only and not fallback_older:
    print("\n→ Fallback was NEVER used. Safe to remove it.")
else:
    print(f"\n→ Fallback was used {len(fallback_only) + len(fallback_older)} times out of {LIMIT}. Consider keeping it.")
