import sqlite3

conn = sqlite3.connect('scrape_cache.db')
cursor = conn.cursor()

# Check total products
cursor.execute('SELECT COUNT(*) FROM product_data')
total = cursor.fetchone()[0]

# Check products with long descriptions
cursor.execute('SELECT COUNT(*) FROM product_data WHERE long_description IS NOT NULL AND long_description != ""')
with_desc = cursor.fetchone()[0]

# Check for a specific sample of spacing issues
cursor.execute('SELECT COUNT(*) FROM product_data WHERE long_description LIKE "%mathandlanguage%"')
spacing_issues = cursor.fetchone()[0]

print(f"Total products: {total}")
print(f"Products with long descriptions: {with_desc}")
print(f"Products with 'mathandlanguage' spacing issue: {spacing_issues}")

conn.close()
