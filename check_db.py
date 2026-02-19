import sqlite3

c = sqlite3.connect("scrape_cache_sel_simple.db")
rows = c.execute("""
    SELECT oldest_review_date, COUNT(*)
    FROM product_metadata
    GROUP BY oldest_review_date
    ORDER BY COUNT(*) DESC
    LIMIT 20
""").fetchall()

print(f"{'value':>30s}  {'count':>6s}")
print("-" * 40)
for val, cnt in rows:
    print(f"{str(val):>30s}  {cnt:>6d}")

total = c.execute("SELECT COUNT(*) FROM product_metadata").fetchone()[0]
print(f"\nTotal rows: {total}")
c.close()
