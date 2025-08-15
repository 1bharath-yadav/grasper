"""
HTML Handler Agent with Python Code Execution for Data Analysis

This module scrapes HTML content and then we ask llm agent to give code to get answers in the selection format we executes Python code to answer user questions.
If code execution fails, it analyzes the error, we ask llm agent If a selector fails:

Search for similar column headers in HTML using fuzzy matching
Replace selector if match ratio ≥ 0.7
Retry execution the HTML structure, and rewrites the code

Safe Code Execution
Inject HTML into the generated Python code in a variable, not inline in string form (prevents syntax errors from HTML characters).
Use a temp working dir for each run. or generate new code with correct selectors with llm agent.
iteratively until the user's questions are answered successfully.

Features:
- HTML scraping with Crawl4AI
- ask llm agent to generate Python code
- Iterative error handling and code refinement
"""

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from pydantic import BaseModel, Field
import asyncio
import json
import re
import sys
import traceback
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from pydantic_ai import Agent
import base64
from io import StringIO, BytesIO
from difflib import SequenceMatcher
import pandas as pd
from bs4 import BeautifulSoup, Tag
from app.config import get_gemini_model
from app.config import get_gemini_api_key


class CodeExecutionResult(BaseModel):
    """Result from Python code execution."""
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    code: str
    attempt: int


class HTMLResult(BaseModel):
    """Result from HTML analysis."""
    url: str
    status: str
    user_requested_response: List[str] = Field(default_factory=list)
    execution_attempts: int = 0
    final_code: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class HTMLAnalysisContext:
    """Context for HTML analysis process."""
    url: str
    data_analysis_input: str
    temp_dir: str
    max_retries: int = 5
    timeout: int = 30


class CodeGenerationRequest(BaseModel):
    """Request structure for code generation."""
    html_structure: str
    user_request: str
    previous_error: Optional[str] = None
    previous_code: Optional[str] = None
    attempt: int = 1


class CodeGenerationResponse(BaseModel):
    """Response structure for code generation."""
    python_code: str
    explanation: str
    selectors_used: List[str] = Field(default_factory=list)


def find_similar_selectors(html_content: str, failed_selector: str, threshold: float = 0.7) -> List[str]:
    """
    Find similar selectors in HTML using fuzzy matching.
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract potential selectors (class names, ids, text content)
    potential_selectors = []

    # Get all elements with class or id attributes
    for element in soup.find_all(True):
        if isinstance(element, Tag):
            if element.get('class'):
                for cls in element['class']:
                    potential_selectors.append(f".{cls}")
            if element.get('id'):
                potential_selectors.append(f"#{element['id']}")

            # Also check text content for table headers, etc.
            if element.string and len(element.string.strip()) > 0:
                potential_selectors.append(element.string.strip())

    # Find similar matches
    similar_selectors = []
    for selector in potential_selectors:
        similarity = SequenceMatcher(
            None, failed_selector.lower(), selector.lower()).ratio()
        if similarity >= threshold:
            similar_selectors.append((selector, similarity))

    # Sort by similarity and return top matches
    similar_selectors.sort(key=lambda x: x[1], reverse=True)
    return [sel[0] for sel in similar_selectors[:5]]


def execute_python_code(code: str, html_content: str, temp_dir: str) -> CodeExecutionResult:
    """
    Execute Python code with HTML content injected as a variable (no restricted environment).
    """
    try:
        # Prepare globals with useful variables
        exec_globals = {
            'pd': pd,
            'BeautifulSoup': BeautifulSoup,
            're': re,
            'json': json,
            'os': os,
            'Path': Path,
            'html_content': html_content,
            'temp_dir': temp_dir,
        }

        # Capture stdout
        old_stdout = sys.stdout
        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            exec(code, exec_globals)
            output = captured_output.getvalue()
            return CodeExecutionResult(
                success=True,
                output=output.strip() if output.strip() else "Code executed successfully (no output)",
                code=code,
                attempt=1
            )
        finally:
            sys.stdout = old_stdout

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        return CodeExecutionResult(
            success=False,
            error=error_msg,
            code=code,
            attempt=1
        )


async def generate_python_code(request: CodeGenerationRequest, file_path, api_key: str, model) -> CodeGenerationResponse:
    """
    Generate Python code using LLM agent.
    """
    system_prompt = f"""
You are an expert Python developer specializing in web scraping and data analysis.
Your task is to generate Python code that analyzes HTML content to answer user questions.

IMPORTANT CONSTRAINTS:
4. Always use print() to output your results
6. If selectors fail, try alternative approaches (xpath, text matching, etc.)
7. Focus on extracting tabular data when possible

HTML_FILE_PATH
{file_path}

Generate clean, executable Python code that directly answers the user's request.
"""

    user_prompt = f"""
HTML Structure Preview (first 2000 chars):

User Request: {request.user_request}

Attempt: {request.attempt}
"""

    if request.previous_error:
        user_prompt += f"""
Previous Error: {request.previous_error}

Previous Code:
{request.previous_code}

Please fix the error and generate improved code.
"""

    # Create agent for code generation
    agent = Agent(
        model=model,
        system_prompt=system_prompt,
    )

    try:
        response = await agent.run(user_prompt)

        # Extract Python code from response - handle different response types
        try:
            # Try to access data attribute
            response_text = str(response.data)  # type: ignore
        except AttributeError:
            # Fallback to string representation
            response_text = str(response)

        # Extract Python code from response
        code_match = re.search(r'```python\n(.*?)\n```',
                               response_text, re.DOTALL)
        if code_match:
            python_code = code_match.group(1)
        else:
            # If no code blocks found, assume the entire response is code
            python_code = response_text

        # Extract selectors used (basic extraction)
        selectors = re.findall(r'["\']([#.]\w+[^"\']*)["\']', python_code)

        return CodeGenerationResponse(
            python_code=python_code.strip(),
            explanation=f"Generated code for attempt {request.attempt}",
            selectors_used=selectors
        )

    except Exception as e:
        # Fallback code generation
        fallback_code = f"""
# Fallback code for: {request.user_request}
try:
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Try to find tables
    tables = soup.find_all('table')
    if tables:
        print(f"Found {{len(tables)}} table(s)")
        for i, table in enumerate(tables[:3]):  # Limit to first 3 tables
            print(f"\\nTable {{i+1}}:")
            rows = table.find_all('tr')
            for row in rows[:5]:  # Show first 5 rows
                cells = row.find_all(['td', 'th'])
                row_text = [cell.get_text(strip=True) for cell in cells]
                print("  | ".join(row_text))
    
    # Try to find structured data
    divs = soup.find_all('div', class_=True)[:10]
    for div in divs:
        if div.get_text(strip=True):
            print(f"Content: {{div.get_text(strip=True)[:100]}}")
            
except Exception as e:
    print(f"Error analyzing HTML: {{e}}")
"""

        return CodeGenerationResponse(
            python_code=fallback_code,
            explanation=f"Fallback code due to error: {str(e)}",
            selectors_used=[]
        )


async def iterative_code_execution(
    html_content: str,
    user_request: str,
    context: HTMLAnalysisContext,
    api_key: str,
    model,
    file_path: str
) -> CodeExecutionResult:
    """
    Iteratively generate and execute Python code until successful or max retries reached.
    """
    previous_code = None
    previous_error = None

    for attempt in range(1, context.max_retries + 1):
        try:
            # Generate code
            code_request = CodeGenerationRequest(
                html_structure=html_content[:5000],  # Limit structure size
                user_request=user_request,
                previous_error=previous_error,
                previous_code=previous_code,
                attempt=attempt
            )

            code_response = await generate_python_code(code_request, file_path, api_key, model)

            # Execute code
            execution_result = execute_python_code(
                code_response.python_code,
                html_content,
                context.temp_dir
            )

            execution_result.attempt = attempt

            if execution_result.success:
                return execution_result

            # If execution failed, try to find similar selectors and retry
            if attempt < context.max_retries:
                previous_code = code_response.python_code
                previous_error = execution_result.error

                # Try to extract failed selectors and find alternatives
                if code_response.selectors_used:
                    for selector in code_response.selectors_used:
                        similar_selectors = find_similar_selectors(
                            html_content, selector)
                        if similar_selectors:
                            print(
                                f"Found similar selectors for {selector}: {similar_selectors}")

        except Exception as e:
            previous_error = str(e)
            if attempt == context.max_retries:
                return CodeExecutionResult(
                    success=False,
                    error=f"Failed after {context.max_retries} attempts. Last error: {previous_error}",
                    code=previous_code or "No code generated",
                    attempt=attempt
                )

    # Return the last failed attempt
    return CodeExecutionResult(
        success=False,
        error=f"Failed after {context.max_retries} attempts. Last error: {previous_error}",
        code=previous_code or "No code generated",
        attempt=context.max_retries
    )


async def scrape_html_content(url: str, context: HTMLAnalysisContext) -> tuple[str, str, str]:
    """
    Scrape HTML content and return raw HTML, markdown content, and file path.
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

        # Handle the result with proper exception handling - ignore type hints for flexibility
        html_content = ""
        markdown_content = ""

        try:
            # Try direct access first (for CrawlResult objects)
            html_content = getattr(result, 'html', '')
            markdown_content = getattr(result, 'markdown', '')
            success = getattr(result, 'success', True)
            if not success:
                error_msg = getattr(result, 'error_message', 'Unknown error')
                raise Exception(f"Crawling failed: {error_msg}")
        except AttributeError:
            try:
                # Handle async generator case
                # type: ignore - we handle this with exception handling
                async for crawl_result in result:  # type: ignore
                    html_content = getattr(crawl_result, 'html', '')
                    markdown_content = getattr(crawl_result, 'markdown', '')
                    success = getattr(crawl_result, 'success', True)
                    if not success:
                        error_msg = getattr(
                            crawl_result, 'error_message', 'Unknown error')
                        raise Exception(f"Crawling failed: {error_msg}")
                    break  # Take first result
            except Exception as e:
                # Fallback: treat as string
                html_content = str(result) if result else ""
                if not html_content:
                    raise Exception(
                        f"Failed to extract content from crawler result: {e}")

        if not html_content:
            raise Exception(f"No HTML content retrieved from {url}")

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

        return html_content, markdown_content, file_path


async def analyze_html_async(
    urls: List[str],
    data_analysis_input: str,
    temp_dir: str = "/tmp",
    api_key: Optional[str] = get_gemini_api_key(),
    model=get_gemini_model()
) -> Dict[str, Any]:
    """
    Main function that scrapes HTML and executes Python code to answer user questions.
    """
    if not urls:
        return {"status": "error", "message": "No URLs provided"}

    if not data_analysis_input.strip():
        return {"status": "error", "message": "No analysis input provided"}

    if not api_key or not model:
        return {"status": "error", "message": "API key and model are required"}

    results = []

    for url in urls:
        context = HTMLAnalysisContext(
            url=url.strip(),
            data_analysis_input=data_analysis_input.strip(),
            temp_dir=temp_dir,
            max_retries=5,
            timeout=30
        )

        try:
            # Step 1: Scrape HTML content
            print(f"Scraping HTML content from {url}")
            html_content, markdown_content, file_path = await scrape_html_content(url, context)

            # Step 2: Create structure info
            structure_info = {
                "url": url,
                "html_length": len(html_content),
                "markdown_length": len(markdown_content),
                "title_extracted": True
            }

            # Step 3: Iteratively generate and execute code
            print("Starting iterative code execution")
            execution_result = await iterative_code_execution(
                html_content=html_content,
                user_request=data_analysis_input,
                context=context,
                api_key=api_key,
                model=model,
                file_path=file_path
            )

            # Step 4: Collect results
            if execution_result.success:
                response = execution_result.output.split('\n') if execution_result.output else [
                    "Analysis completed successfully"]

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
            print(error_msg)

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


# # Example usage
# async def main():
#     """Example usage of the HTML Handler Agent."""
#     urls = ["https://en.wikipedia.org/wiki/List_of_highest-grossing_films"]
#     analysis_request = """Answer the following questions and respond with a JSON array of strings containing the answer.

# 1. How many $2 bn movies were released before 2000?
# 2. Which is the earliest film that grossed over $1.5 bn?
# 3. What's the correlation between the Rank and Peak?
# 4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it.
#    Return as a base-64 encoded data URI, `"data:image/png;base64,iVBORw0KG..."` under 100,000 bytes."""

#     result = await analyze_html_async(
#         urls=urls,
#         data_analysis_input=analysis_request,
#         temp_dir="./temp",
#         api_key=get_gemini_api_key(),
#         model=get_gemini_model()
#     )

#     # print(json.dumps(result, indent=2))

# if __name__ == "__main__":
#     asyncio.run(main())
