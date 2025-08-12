from bs4 import BeautifulSoup
import requests
import asyncio
from pathlib import Path
import re
from urllib.parse import urlparse
import sys
import json


def generate_filename_from_url(url: str) -> str:
    parsed = urlparse(url)

    # Combine domain and path
    domain = parsed.netloc.replace('.', '_').replace(':', '_')
    path_parts = parsed.path.strip('/').replace('/', '_').replace('\\', '_')

    # Remove special characters and spaces
    filename_base = f"{domain}_{path_parts}" if path_parts else domain
    filename_base = re.sub(r'[^\w\-_]', '_', filename_base)

    # Remove multiple underscores and trailing underscores
    filename_base = re.sub(r'_+', '_', filename_base).strip('_')

    # Ensure it's not empty and add .html extension
    if not filename_base:
        filename_base = "scraped_page"

    return f"{filename_base}.html"


async def scrape_html(url: str, path) -> dict:
    """
    Scrape HTML content using Playwright for JavaScript-heavy websites.
    """
    try:
        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            # Navigate to the page and wait for it to load
            await page.goto(url, wait_until="networkidle")

            # Wait a bit more for any dynamic content
            await page.wait_for_timeout(2000)

            # Get the fully rendered HTML
            html_content = await page.content()

            await browser.close()

        soup = BeautifulSoup(html_content, "html.parser")

        # Save the HTML content in a separate html directory
        html_dir = Path(path)
        html_dir.mkdir(exist_ok=True)

        filename = generate_filename_from_url(url)
        output_file = html_dir / filename
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(str(soup.prettify()))

        print(f"Successfully scraped HTML content from {url} using Playwright")
        return {"url": url, "output_path": str(output_file), "filename": filename}

    except ImportError:
        print("Playwright not installed. Installing playwright...")
        # Try to install playwright
        import subprocess
        try:
            subprocess.run(["uv", "add", "playwright"], check=True)
            subprocess.run(["playwright", "install", "chromium"], check=True)
            print("Playwright installed successfully. Retrying...")
            return await scrape_html(url, path)
        except Exception as install_error:
            print(f"Failed to install Playwright: {install_error}")
            return {"url": url, "error": f"Failed to scrape {url}: Playwright not available"}

    except Exception as e:
        print(f"Playwright scraping failed for {url}: {e}")
        return {"url": url, "error": f"Failed to scrape {url}: {str(e)}"}
