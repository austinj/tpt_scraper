@echo off
echo Setting up TPT Scraper environment...
python -m venv venv
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install aiohttp aiohttp-client-cache aiosqlite beautifulsoup4 lxml playwright tqdm async-timeout
playwright install
echo Environment ready! Use 'venv\Scripts\activate' to activate.