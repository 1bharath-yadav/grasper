"""
HTML Handler Agent with Python Code Execution for Data Analysis

This module scrapes HTML content and then generates and executes Python code to answer user questions.
If code execution fails, it analyzes the error, examines the HTML structure, and rewrites the code
iteratively until the user's questions are answered successfully.

Features:
- HTML scraping with Crawl4AI
- Dynamic Python code generation based on user requests
- Iterative error handling and code refinement
- HTML structure analysis for better selector targeting
- Support for data analysis, visualizations, and complex queries
- Safe code execution environment
"""

from app.config import get_gemini_model, get_gemini_api_key
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from pydantic import BaseModel, Field
import logfire
import asyncio
import json
import re
import sys
import traceback
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from pydantic_ai import Agent
import base64
from io import StringIO, BytesIO


@dataclass
class HTMLAnalysisContext:
    """Context for HTML analysis operations."""
    url: str
    data_analysis_input: str
    temp_dir: str
    max_retries: int = 5
    timeout: int = 30


class CodeExecutionResult(BaseModel):
    """Result from Python code execution."""
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    code: str
    attempt: int


class HTMLResult(BaseModel):
    """Final result from HTML analysis."""
    url: str
    status: str
    user_requested_response: Union[List[str], List[Any], Dict[str, Any]]
    execution_attempts: int = 0
    final_code: Optional[str] = None
    error_message: Optional[str] = None


async def scrape_html_content(url: str, context: HTMLAnalysisContext) -> tuple[str, str]:
    """
    Scrape HTML content and return both raw HTML and cleaned content.
    """
    browser_config = BrowserConfig(
        headless=True,
        java_script_enabled=True,
        browser_type="chromium",
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        crawl_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            page_timeout=context.timeout * 1000,
            wait_for_images=False,
            screenshot=False,
            verbose=True
        )

        result = await crawler.arun(url=url, config=crawl_config)

        if not result.success:
            raise Exception(f"Failed to scrape {url}: {result.error_message}")

        html_content = result.html
        # Extract title from HTML
        title_match = re.search(r'<title>(.*?)</title>',
                                html_content, re.IGNORECASE | re.DOTALL)
        if title_match:
            title = title_match.group(1).strip()
            # Clean title for filename (remove illegal/special chars)
            title = re.sub(r'[^\w\-\_\. ]', '', title)
            if not title:
                title = 'scraped_page'
        else:
            title = 'scraped_page'

        filename = f"{title}.html"
        file_path = os.path.join(context.temp_dir, filename)
        # Ensure temp_dir exists
        os.makedirs(context.temp_dir, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return html_content, result.markdown


def analyze_html_structure(html_content: str) -> Dict[str, Any]:
    """
    Analyze HTML structure to provide context for code generation.
    """
    structure_info = {
        "has_tables": bool(re.search(r'<table[^>]*>', html_content, re.IGNORECASE)),
        "table_count": len(re.findall(r'<table[^>]*>', html_content, re.IGNORECASE)),
        "has_lists": bool(re.search(r'<[uo]l[^>]*>', html_content, re.IGNORECASE)),
        "list_count": len(re.findall(r'<[uo]l[^>]*>', html_content, re.IGNORECASE)),
        "common_classes": [],
        "common_ids": [],
        "table_classes": [],
        "content_length": len(html_content)
    }

    # Extract common classes
    classes = re.findall(r'class=["\']([^"\']*)["\']', html_content)
    all_classes = [c for cls in classes for c in cls.split()]
    class_counts = {}
    for cls in all_classes:
        class_counts[cls] = class_counts.get(cls, 0) + 1
    structure_info["common_classes"] = sorted(
        class_counts.keys(), key=class_counts.get, reverse=True)[:20]

    # Extract IDs
    ids = re.findall(r'id=["\']([^"\']*)["\']', html_content)
    structure_info["common_ids"] = list(set(ids))[:20]

    # Extract table-specific classes
    table_matches = re.findall(
        r'<table[^>]*class=["\']([^"\']*)["\'][^>]*>', html_content, re.IGNORECASE)
    structure_info["table_classes"] = [cls.strip()
                                       for match in table_matches for cls in match.split()]

    return structure_info


async def generate_python_code(
    html_content: str,
    user_request: str,
    structure_info: Dict[str, Any],
    previous_error: Optional[str] = None,
    previous_code: Optional[str] = None,
    attempt: int = 1
) -> str:
    """
    Generate Python code to extract data and answer user questions.
    """
    gemini_model = get_gemini_model()
    agent = Agent(
        model=gemini_model,
        system_prompt=(
            "You are an expert Python programmer specializing in web scraping and data analysis. "
            "Write Python code that parses HTML content to extract data and answer user questions. "
            "Use BeautifulSoup for HTML parsing, pandas for data manipulation, matplotlib/seaborn for plotting, "
            "and any other necessary libraries. The code should be complete and executable."
        )
    )

    error_context = ""
    if previous_error and previous_code:
        error_context = f"""
        PREVIOUS ATTEMPT FAILED:
        Previous Code:
        ```python
        {previous_code}
        ```
        
        Error Encountered:
        {previous_error}
        
        Please fix the error and rewrite the code. Pay attention to:
        - HTML structure and available CSS selectors
        - Data types and format conversions
        - Missing imports or dependencies
        - Correct element selection strategies
        """

    html_sample = html_content[:5000] if len(
        html_content) > 5000 else html_content

    prompt = f"""
    Write Python code to scrape and analyze HTML content to answer the user's request.
    
    USER REQUEST: {user_request}
    
    HTML STRUCTURE ANALYSIS:
    {json.dumps(structure_info, indent=2)}
    
    HTML SAMPLE (first 5000 chars):
    {html_sample}
    
    {error_context}
    
    REQUIREMENTS:
    1. The HTML content is available in a variable called 'html_content'
    2. Write complete, executable Python code
    3. Use BeautifulSoup for HTML parsing: from bs4 import BeautifulSoup
    4. Use pandas for data manipulation if needed
    5. For plots, save as base64 encoded PNG and return as data URI
    6. Print or return results in the exact format requested by the user
    7. Handle edge cases and data cleaning
    8. If user wants JSON array of strings, ensure that format is returned
    9. Include all necessary imports
    10. Add error handling where appropriate
    
    ATTEMPT #{attempt}
    
    Return only the Python code, no explanations.
    """

    result = await agent.run(prompt)
    code = result.data if hasattr(result, 'data') else str(result)

    # Clean the code (remove markdown formatting if present)
    if '```python' in code:
        code = code.split('```python')[1].split('```')[0]
    elif '```' in code:
        code = code.split('```')[1].split('```')[0]

    return code.strip()


def execute_python_code(code: str, html_content: str, temp_dir: str) -> CodeExecutionResult:
    """
    Execute Python code with HTML content in a controlled environment.
    """
    try:
        # Create a temporary file to hold the code
        temp_file = Path(temp_dir) / f"temp_code_{os.urandom(8).hex()}.py"

        # Prepare the code with HTML content injection

        # Write code to temporary file
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(full_code)

        # Execute the code and capture output
        result = subprocess.run(
            [sys.executable, temp_file],
            cwd=temp_dir,
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout
        )

        # Clean up
        try:
            os.remove(temp_file)
        except:
            pass

        if result.returncode == 0:
            return CodeExecutionResult(
                success=True,
                output=result.stdout.strip(),
                code=code,
                attempt=1
            )
        else:
            return CodeExecutionResult(
                success=False,
                error=result.stderr.strip(),
                code=code,
                attempt=1
            )

    except subprocess.TimeoutExpired:
        return CodeExecutionResult(
            success=False,
            error="Code execution timed out after 60 seconds",
            code=code,
            attempt=1
        )
    except Exception as e:
        return CodeExecutionResult(
            success=False,
            error=f"Execution failed: {str(e)}",
            code=code,
            attempt=1
        )


async def iterative_code_execution(
    html_content: str,
    user_request: str,
    structure_info: Dict[str, Any],
    context: HTMLAnalysisContext
) -> CodeExecutionResult:
    """
    Iteratively generate and execute Python code until successful or max retries reached.
    """
    previous_code = None
    previous_error = None

    for attempt in range(1, context.max_retries + 1):
        with logfire.span("code_generation_attempt", attempt=attempt):
            try:
                # Generate Python code
                code = await generate_python_code(
                    html_content=html_content,
                    user_request=user_request,
                    structure_info=structure_info,
                    previous_error=previous_error,
                    previous_code=previous_code,
                    attempt=attempt
                )

                logfire.info("Generated code", attempt=attempt,
                             code_length=len(code))

                # Execute the code
                result = execute_python_code(
                    code, html_content, context.temp_dir)
                result.attempt = attempt

                if result.success:
                    logfire.info("Code execution successful", attempt=attempt)
                    return result

                # Log the error and prepare for next attempt
                logfire.warn("Code execution failed",
                             attempt=attempt, error=result.error)
                previous_code = code
                previous_error = result.error

                # Add more specific error handling for common issues
                if attempt < context.max_retries:
                    await asyncio.sleep(1)  # Brief pause between attempts

            except Exception as e:
                logfire.error("Code generation failed",
                              attempt=attempt, error=str(e))
                previous_error = f"Code generation error: {str(e)}"

    # Return the last failed attempt
    return CodeExecutionResult(
        success=False,
        error=f"Failed after {context.max_retries} attempts. Last error: {previous_error}",
        code=previous_code or "No code generated",
        attempt=context.max_retries
    )


async def analyze_html_async(urls: List[str], data_analysis_input: str, temp_dir: str = "/tmp") -> Dict[str, Any]:
    """
    Main function that scrapes HTML and executes Python code to answer user questions.
    """
    if not urls:
        return {"status": "error", "message": "No URLs provided"}

    if not data_analysis_input.strip():
        return {"status": "error", "message": "No analysis input provided"}

    results = []

    for url in urls:
        with logfire.span("analyze_html_with_code_execution", url=url):
            context = HTMLAnalysisContext(
                url=url.strip(),
                data_analysis_input=data_analysis_input.strip(),
                temp_dir=temp_dir,
                max_retries=5,
                timeout=30
            )

            try:
                # Normalize URL
                if not url.startswith(('http://', 'https://')):
                    url = 'https://' + url
                    context.url = url

                # Step 1: Scrape HTML content
                logfire.info("Scraping HTML content", url=url)
                html_content, markdown_content = await scrape_html_content(url, context)

                # Step 2: Analyze HTML structure
                logfire.info("Analyzing HTML structure")
                structure_info = analyze_html_structure(html_content)

                # Step 3: Iteratively generate and execute code
                logfire.info("Starting iterative code execution")
                execution_result = await iterative_code_execution(
                    html_content=html_content,
                    user_request=data_analysis_input,
                    structure_info=structure_info,
                    context=context
                )

                if execution_result.success:
                    # Parse the output to get the user requested response
                    output = execution_result.output

                    # Try to parse as JSON if it looks like JSON
                    try:
                        if output.startswith('[') and output.endswith(']'):
                            response = json.loads(output)
                        elif output.startswith('{') and output.endswith('}'):
                            response = json.loads(output)
                        else:
                            # Split by lines for multiple answers
                            response = [line.strip() for line in output.split(
                                '\n') if line.strip()]
                    except json.JSONDecodeError:
                        response = [output] if output else [
                            "No output generated"]

                    results.append(HTMLResult(
                        url=url,
                        status="success",
                        user_requested_response=response,
                        execution_attempts=execution_result.attempt,
                        final_code=execution_result.code
                    ).dict())
                else:
                    results.append(HTMLResult(
                        url=url,
                        status="error",
                        user_requested_response=[
                            f"Code execution failed: {execution_result.error}"],
                        execution_attempts=execution_result.attempt,
                        final_code=execution_result.code,
                        error_message=execution_result.error
                    ).dict())

            except Exception as e:
                error_msg = f"Analysis failed for {url}: {str(e)}"
                logfire.error("HTML analysis failed", error=str(e), url=url)

                results.append(HTMLResult(
                    url=url,
                    status="error",
                    user_requested_response=[error_msg],
                    error_message=str(e)
                ).dict())

    # Return single result for single URL, aggregated results for multiple URLs
    if len(urls) == 1:
        return results[0] if results else {"status": "error", "message": "Analysis failed"}

    return {
        "status": "success" if any(r["status"] == "success" for r in results) else "error",
        "results": results,
        "summary": {
            "total_urls": len(urls),
            "successful": len([r for r in results if r["status"] == "success"]),
            "failed": len([r for r in results if r["status"] == "error"])
        }
    }


def analyze_html(urls: List[str], data_analysis_input: str, temp_dir: str) -> Dict[str, Any]:
    """
    Synchronous wrapper for async HTML analysis with code execution.
    """
    try:
        # Validate inputs
        if not isinstance(urls, list):
            urls = [urls] if urls else []

        urls = [url for url in urls if url and isinstance(url, str)]

        if not urls:
            return {"status": "error", "message": "No valid URLs provided"}

        if not data_analysis_input or not isinstance(data_analysis_input, str):
            return {"status": "error", "message": "Valid data_analysis_input is required"}

        # Ensure temp directory exists
        os.makedirs(temp_dir, exist_ok=True)

        # Run async analysis
        return asyncio.run(analyze_html_async(urls, data_analysis_input, temp_dir))

    except Exception as e:
        error_msg = f"HTML analysis wrapper failed: {str(e)}"
        logfire.error("HTML analysis wrapper failed", error=str(e))
        return {
            "status": "error",
            "message": error_msg,
            "url": urls[0] if urls else "unknown"
        }
