#!/usr/bin/env python3
"""
Test script to verify HTML analyzer integration
"""

import sys
import json
from pathlib import Path

# Add app directory to path
sys.path.append(str(Path(__file__).parent / "app"))


def test_html_analyzer():
    """Test the HTML analyzer integration"""
    try:
        from app.agent.html_analyzer import analyze_html_default

        # Test with the existing HTML file
        html_file = Path(
            "tmp/5bc8dac704/en_wikipedia_org_wiki_List_of_highest-grossing_films.html")

        if not html_file.exists():
            print(f"Test HTML file not found: {html_file}")
            return False

        print(f"Testing HTML analyzer with: {html_file}")

        # Run analysis
        result = analyze_html_default(html_file)

        # Print results
        print("\n=== HTML Analysis Results ===")
        print(f"Title: {result.get('title', 'N/A')}")
        print(f"Description: {result.get('description', 'N/A')}")
        print(f"Number of headings: {result.get('headings_count', 0)}")
        print(f"Number of forms: {result.get('forms_count', 0)}")
        print(f"Number of links: {result.get('links_count', 0)}")
        print(f"Number of images: {result.get('images_count', 0)}")

        if result.get('error'):
            print(f"Error: {result['error']}")
            return False

        print("\n✅ HTML analyzer integration test passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_html_analyzer()
    sys.exit(0 if success else 1)
