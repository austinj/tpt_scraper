import sqlite3

conn = sqlite3.connect('scrape_cache_sel_simple.db')
c = conn.cursor()

# List all tables
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in c.fetchall()]
print(f'Tables: {tables}')

for t in tables:
    c.execute(f'SELECT COUNT(*) FROM [{t}]')
    print(f'  {t}: {c.fetchone()[0]:,} rows')

# Check product_metadata
c.execute('SELECT COUNT(*) FROM product_metadata')
total = c.fetchone()[0]
print(f'\nTotal products: {total:,}')

c.execute("SELECT COUNT(*) FROM product_metadata WHERE oldest_review_date IS NULL")
null_count = c.fetchone()[0]
print(f'NULL oldest_review_date: {null_count:,}')

c.execute("SELECT COUNT(*) FROM product_metadata WHERE oldest_review_date = 'no_reviews'")
no_rev = c.fetchone()[0]
print(f'no_reviews: {no_rev:,}')

c.execute("SELECT COUNT(*) FROM product_metadata WHERE oldest_review_date = 'parse_failed'")
pf = c.fetchone()[0]
print(f'parse_failed: {pf:,}')

c.execute("SELECT COUNT(*) FROM product_metadata WHERE oldest_review_date NOT IN ('no_reviews', 'parse_failed') AND oldest_review_date IS NOT NULL")
dated = c.fetchone()[0]
print(f'Has actual date: {dated:,}')

c.execute("SELECT COUNT(*) FROM product_metadata WHERE CAST(number_of_ratings AS INTEGER) > 0 AND oldest_review_date IS NULL")
need_scrape = c.fetchone()[0]
print(f'\nNeed deepscrape (has ratings, no date): {need_scrape:,}')

conn.close()
