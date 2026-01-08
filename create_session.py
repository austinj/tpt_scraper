"""
Generate tpt_storage.json by logging into TPT manually.
Run this script, log in when the browser opens, then close the browser.
"""
import asyncio
from playwright.async_api import async_playwright

async def save_session():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # Visible browser
        context = await browser.new_context()
        page = await context.new_page()
        
        # Go to TPT login
        await page.goto("https://www.teacherspayteachers.com/Login")
        
        print("\n" + "="*60)
        print("LOG IN TO TPT IN THE BROWSER WINDOW")
        print("Then close the browser when you're logged in.")
        print("="*60 + "\n")
        
        # Wait for user to log in and close
        try:
            await page.wait_for_event("close", timeout=300000)  # 5 min timeout
        except:
            pass
        
        # Save session state
        await context.storage_state(path="tpt_storage.json")
        print("\n✅ Session saved to tpt_storage.json")
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(save_session())
