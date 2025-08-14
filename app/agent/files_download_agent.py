from app.config import get_gemini_model
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


# Initialize model from config
model = get_gemini_model()


class DataToolset(AbstractToolset[AgentDeps]):
    async def __aenter__(self):
        return self

    async def get_tools(self, ctx):
        return {}

    async def call_tool(self, name: str, tool_args: dict, ctx, tool):
        raise NotImplementedError("DataToolset does not support tool calling")


async def handle_urls(data_analyst_input: str, temp_dir: str) -> List[str]:
    # Early exit if no URLs found
    if not data_analyst_input or ("http://" not in data_analyst_input and "https://" not in data_analyst_input):
        print("[DEBUG] No URLs found in input. Skipping processing.")
        return []

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

    # If no URLs in both categories, skip further processing
    if not url_data.website_urls and not url_data.download_urls:
        print("[DEBUG] Classification found no URLs. Skipping processing.")
        return []

    # Step 2: Handle website URLs with html_scraper

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
        downloaded_files=url_data.download_urls,
        temp_dir=temp_dir
    )

    download_result = await download_agent.run(
        f"Download and validate files from: {url_data.download_urls}",
        deps=deps
    )

    all_files = download_result.output.total_files
    print(f"[DEBUG] Total files after download: {len(all_files)}")
    return list(set(all_files))
