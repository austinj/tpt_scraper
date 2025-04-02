import os
import shutil
import aiosqlite
import asyncio
import pandas as pd
from pathlib import Path
import xlsxwriter
from tqdm import tqdm

DB_FILE = "scrape_cache.db"
EXPORT_FOLDER = Path("free_product_export")
FILES_FOLDER = EXPORT_FOLDER / "files"
EXCEL_FILE = EXPORT_FOLDER / "free_products.xlsx"

EXPORT_FOLDER.mkdir(exist_ok=True)
FILES_FOLDER.mkdir(exist_ok=True)

async def export_to_excel():
    async with aiosqlite.connect(DB_FILE) as db:
        query = """
        SELECT
            p.title,
            p.short_description,
            p.long_description,
            p.rating_value,
            p.number_of_ratings,
            p.product_price,
            p.url,
            p.folder,
            p.price_option,
            p.sort_order,
            p.page,
            d.free_file_path
        FROM product_data p
        JOIN free_file_downloads d ON p.id = d.product_id
        """
        async with db.execute(query) as cursor:
            rows = await cursor.fetchall()

    columns = [
        "Title",
        "Short Description",
        "Long Description",
        "Rating",
        "Number of Ratings",
        "Price",
        "Product URL",
        "Folder",
        "Price Option",
        "Sort Order",
        "Page",
        "Original File Path"
    ]
    df = pd.DataFrame(rows, columns=columns)

    # Copy files and build relative file links
    cleaned_rows = []
    hyperlink_paths = []

    print("📦 Copying files and building Excel rows...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing products"):
        original_path = Path(row["Original File Path"])
        if original_path.exists():
            dest_path = FILES_FOLDER / original_path.name
            shutil.copy2(original_path, dest_path)
            hyperlink_paths.append(dest_path.relative_to(EXPORT_FOLDER).as_posix())
            cleaned_rows.append(row)
        else:
            print(f"⚠️ Skipping missing file: {original_path}")

    # Only keep rows with existing files
    df = pd.DataFrame(cleaned_rows)
    df["File Link"] = hyperlink_paths
    df.drop(columns=["Original File Path"], inplace=True)

    # Write to Excel with hyperlinks
    with pd.ExcelWriter(EXCEL_FILE, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Free Products', startrow=1, header=False)
        workbook = writer.book
        worksheet = writer.sheets['Free Products']

        header_format = workbook.add_format({'bold': True, 'bg_color': '#eeeeee'})
        for col_num, value in enumerate(df.columns):
            worksheet.write(0, col_num, value, header_format)

        # Apply hyperlinks to file link column
        file_col_idx = df.columns.get_loc("File Link")
        for row_num, link in enumerate(df["File Link"], start=1):
            worksheet.write_url(row_num, file_col_idx, link, string=Path(link).name)

    print(f"✅ Export complete: {EXCEL_FILE.resolve()}")

if __name__ == "__main__":
    asyncio.run(export_to_excel())
