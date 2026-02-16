import os
from typing import List
from dataclasses import dataclass
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider
import logfire
from pydantic_ai.toolsets.abstract import AbstractToolset
from app.agent.html_handler import scrape_html

load_dotenv()
logfire.configure()


@dataclass
class AgentDeps:
    data_analyst_input: str
    downloaded_files: List[str]
    temp_dir: str


class UrlClassification(BaseModel):
    website_urls: List[str]
    download_urls: List[str]


class DataPlanSchema(BaseModel):
    total_files: List[str]


# Initialize model
gemini_key = os.getenv("GEMINI_API_KEY")
if not gemini_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

settings = GoogleModelSettings(google_thinking_config={"thinking_budget": 0})
provider = GoogleProvider(api_key=gemini_key)
model = GoogleModel("gemini-2.5-flash", provider=provider, settings=settings)


class DataToolset(AbstractToolset[AgentDeps]):
    async def __aenter__(self):
        return self

    async def get_tools(self, ctx):
        return {}

    async def call_tool(self, name: str, tool_args: dict, ctx, tool):
        raise NotImplementedError("DataToolset does not support tool calling")


async def handle_data(data_analyst_input: str, temp_dir: str) -> List[str]:
    # Step 1: Classify URLs
    classify_agent = Agent(
        model=model,
        output_type=UrlClassification,
        system_prompt=(
            "Parse the input text and extract all URLs. Classify them into two categories:\n"
            "1. website_urls: HTTP/HTTPS URLs that point to web pages (html content)\n"
            "2. download_urls: URLs that point to downloadable files (images, videos, audio, csv, excel, zip, pdf, etc.)\n"
            "Return the classification in the specified format."
        )
    )

    classification = await classify_agent.run(f"Classify URLs from: {data_analyst_input}")
    url_data = classification.output

    # Step 2: Handle website URLs with html_scraper
    scraped_files = []
    for url in url_data.website_urls:
        try:
            result = await scrape_html(url=url, path=temp_dir)
            # Collect the output_path if scraping was successful
            if result and isinstance(result, dict) and 'output_path' in result:
                scraped_files.append(result['output_path'])
            print(
                f"[DEBUG] Scraped file from {url}: {result.get('output_path')}")
        except Exception as e:
            print(f"Error scraping {url}: {e}")

    # Step 3: Generate and execute download code for file URLs
    download_agent = Agent(
        model=model,
        deps_type=AgentDeps,
        output_type=DataPlanSchema,
        system_prompt=(
            "Generate Python code to download files from the provided URLs to {temp_dir}. "
            "Use requests library. Validate downloads and retry if corrupted. "
            "Handle different file types appropriately. Return all successfully downloaded file paths."
        ),
        retries=3,
        toolsets=[DataToolset()]
    )

    deps = AgentDeps(
        data_analyst_input=f"Download files from URLs: {url_data.download_urls}",
        downloaded_files=scraped_files,
        temp_dir=temp_dir
    )

    download_result = await download_agent.run(
        f"Download and validate files from: {url_data.download_urls}",
        deps=deps
    )

    all_files = scraped_files + download_result.output.total_files
    print(f"[DEBUG] Total files after download: {len(all_files)}")
    return list(set(all_files))  # Remove duplicates
