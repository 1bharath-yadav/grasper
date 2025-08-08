from bs4 import BeautifulSoup
import requests
import asyncio
from pathlib import Path
import re
from urllib.parse import urlparse


def generate_filename_from_url(url: str) -> str:
    """
    Generate a safe filename from a URL.
    Examples:
    - https://example.com/page -> example_com_page.html
    - https://api.example.com/v1/data -> api_example_com_v1_data.html
    """
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
    Scrape HTML content from a URL. 
    First tries requests, then falls back to Playwright for JavaScript-heavy sites.
    """
    try:
        # First attempt: Use requests for simple HTML scraping
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Check if there's meaningful content (not just scripts/empty)
        text_content = soup.get_text(strip=True)

        if len(text_content) < 100:  # Very little content, might be JavaScript-heavy
            print(
                f"Little content found with requests, trying Playwright for {url}")
            return await scrape_with_playwright(url, path)

        # Save the HTML content in a separate html directory
        html_dir = Path(path) / "html"
        html_dir.mkdir(exist_ok=True)

        filename = generate_filename_from_url(url)
        output_file = html_dir / filename
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(str(soup.prettify()))

        print(f"Successfully scraped HTML content from {url} using requests")
        return {"url": url, "output_path": str(output_file), "method": "requests", "filename": filename}

    except Exception as e:
        print(f"Requests failed for {url}: {e}. Trying Playwright...")
        return await scrape_with_playwright(url, path)


async def scrape_with_playwright(url: str, path) -> dict:
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
        html_dir = Path(path) / "html"
        html_dir.mkdir(exist_ok=True)

        filename = generate_filename_from_url(url)
        output_file = html_dir / filename
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(str(soup.prettify()))

        print(f"Successfully scraped HTML content from {url} using Playwright")
        return {"url": url, "output_path": str(output_file), "method": "playwright", "filename": filename}

    except ImportError:
        print("Playwright not installed. Installing playwright...")
        # Try to install playwright
        import subprocess
        try:
            subprocess.run(["pip", "install", "playwright"], check=True)
            subprocess.run(["playwright", "install", "chromium"], check=True)
            print("Playwright installed successfully. Retrying...")
            return await scrape_with_playwright(url, path)
        except Exception as install_error:
            print(f"Failed to install Playwright: {install_error}")
            return {"url": url, "error": f"Failed to scrape {url}: Playwright not available"}

    except Exception as e:
        print(f"Playwright scraping failed for {url}: {e}")
        return {"url": url, "error": f"Failed to scrape {url}: {str(e)}"}
