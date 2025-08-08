#!/usr/bin/env python3
from app.agent.html_handler import scrape_html
import asyncio
from pathlib import Path
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def test_html_scraping():
    test_dir = Path("/tmp/test_html_scraping")
    test_dir.mkdir(exist_ok=True)

    # Test with a simple webpage
    print("Testing HTML scraping with httpbin.org...")
    result = await scrape_html("https://httpbin.org/html", test_dir)
    print(f"Result: {result}")

    # Check if file was created
    if "output_path" in result:
        output_file = Path(result["output_path"])
        if output_file.exists():
            print(f"✅ HTML file created: {output_file}")
            print(f"File size: {output_file.stat().st_size} bytes")

            # Show first few lines
            with open(output_file, 'r') as f:
                content = f.read()[:500]
                print(f"Content preview:\n{content}...")
        else:
            print("❌ HTML file not found")
    else:
        print(f"❌ Error in scraping: {result}")

if __name__ == "__main__":
    asyncio.run(test_html_scraping())
