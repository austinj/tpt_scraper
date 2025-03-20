import sqlite3
import csv

DB_FILE = "scrape_cache.db"

def export_table_to_csv(table_name, output_file):
    # Connect to the database
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Retrieve column names from the table
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns_info = cursor.fetchall()
    column_names = [info[1] for info in columns_info]
    
    # Retrieve all rows from the table
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()
    
    # Write the data to CSV
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(column_names)  # write header
        writer.writerows(rows)         # write data rows
    
    conn.close()
    print(f"Exported '{table_name}' to '{output_file}'.")

if __name__ == "__main__":
    # Export both tables to CSV files
    export_table_to_csv("extracted_urls", "extracted_urls.csv")
    export_table_to_csv("product_data", "product_data.csv")
