$py = ".\.venv\Scripts\python.exe"
$config = "coding"

Write-Host "=== TPT Free Products Pipeline ===" -ForegroundColor Cyan

# Session check
if (-not (Test-Path "tpt_storage.json")) {
    Write-Host "`nNo session file found. Opening browser to log in..." -ForegroundColor Yellow
    & $py create_session.py
    if ($LASTEXITCODE -ne 0) { Write-Host "Session creation failed. Exiting." -ForegroundColor Red; exit 1 }
}

# Search
Write-Host "`n[1/3] Searching for free products..." -ForegroundColor Cyan
& $py tpt_scraper.py search $config
if ($LASTEXITCODE -ne 0) { Write-Host "Search failed. Exiting." -ForegroundColor Red; exit 1 }

# Scrape
Write-Host "`n[2/3] Scraping metadata..." -ForegroundColor Cyan
& $py tpt_scraper.py scrape $config
if ($LASTEXITCODE -ne 0) { Write-Host "Scrape failed. Exiting." -ForegroundColor Red; exit 1 }

# Stats
Write-Host "`n--- Stats after scrape ---" -ForegroundColor DarkCyan
& $py tpt_scraper.py stats $config

# Download
Write-Host "`n[3/3] Downloading top 500 by rating count..." -ForegroundColor Cyan
& $py tpt_scraper.py download $config --top 500
if ($LASTEXITCODE -ne 0) { Write-Host "Download failed or session expired. Re-run create_session.py then run this script again." -ForegroundColor Red; exit 1 }

# Final stats
Write-Host "`n--- Final stats ---" -ForegroundColor DarkCyan
& $py tpt_scraper.py stats $config

Write-Host "`nDone. Files in: downloads_$config\" -ForegroundColor Green
