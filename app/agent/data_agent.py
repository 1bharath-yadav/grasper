from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict
import hashlib
import aiofiles
import asyncio
import os

from dotenv import load_dotenv
from pydantic import BaseModel, RootModel
from pydantic_ai import Agent, RunContext, StructuredDict, ModelRetry
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider
import logfire

from app.agent.executor import create_temp_dir
from app.agent.html_handler import scrape_html
from .utils import convert_numpy_types

load_dotenv()

# Configure Logfire
logfire.configure()

load_dotenv()

# ---- Agent Dependency Types ----


@dataclass
class AgentDeps:
    data_analyst_input: str
    downloaded_files: List[str]
    temp_dir: Optional[str] = None

# ---- Data Model Definitions ----


class DataSourcePlan(BaseModel):
    sourcename: str
    data_download_pythonic_code: str
    url: Optional[str] = None
    html: bool = False


class DataPlan(RootModel[dict[str, DataSourcePlan]]):
    root: dict[str, DataSourcePlan]


data_plan_schema = {
    "type": "object",
    "additionalProperties": {
        "type": "object",
        "properties": {
            "sourcename": {"type": "string"},
            "data_download_pythonic_code": {"type": "string"},
            "url": {"type": "string"},
            "html": {"type": "boolean"}
        },
        "required": ["sourcename", "data_download_pythonic_code"],
    }
}

# ---- LLM Model Setup ----

gemini_key = os.getenv("GEMINI_API_KEY")
if not gemini_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

settings = GoogleModelSettings(google_thinking_config={"thinking_budget": 0})
provider = GoogleProvider(api_key=gemini_key)
model = GoogleModel("gemini-2.5-flash", provider=provider)

# ---- Agent Initialization ----

agent = Agent(
    model,
    deps_type=AgentDeps,
    output_type=StructuredDict(data_plan_schema),
    system_prompt=(
        "You are a data sourcing agent. Your ONLY job is to download files from URLs. "
        "Given URLs/instructions in `data_analyst_input`, provide Python code that ONLY downloads files to {tmp} directory. "
        "DO NOT include any data analysis, processing, plotting, or calculations. "
        "ONLY use requests.get() to download files and save them with appropriate names. "
        "Example: requests.get(url) -> save to {tmp}/filename.csv "
        "EXCEPTION - Don't download HTML sources."
    ),
    retries=3,
)

# ---- In-Memory Cache ----

input_plan_cache: Dict[str, DataPlan] = {}

# ---- Tool Definitions ----


@agent.tool
async def html_tool(ctx: RunContext[AgentDeps], url: str) -> dict:
    """Fetches and scrapes HTML from the given URL into the temp directory."""
    temp_dir = await ensure_temp_dir(ctx.deps)
    return await scrape_html(url, Path(temp_dir))


@agent.tool
async def code_tool(
    ctx: RunContext[AgentDeps],
    scraping_code: str,
    sourcename: str
) -> dict:
    """Runs provided scraping code and validates files are > 1KB. Installs missing modules if needed."""
    deps = ctx.deps
    temp_dir_path = Path(await ensure_temp_dir(deps))

    # Replace {tmp} placeholder with actual temp directory path
    scraping_code = scraping_code.replace("{tmp}", str(temp_dir_path))

    # Write scrape script
    script_path = temp_dir_path / f"{sourcename}_scrape.py"
    async with aiofiles.open(script_path, "w") as f:
        await f.write(scraping_code)

    async def run_script():
        proc = await asyncio.create_subprocess_exec(
            "python", str(script_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        return proc.returncode, stdout, stderr

    # First attempt
    returncode, stdout, stderr = await run_script()
    if returncode != 0:
        error_msg = stderr.decode().strip()

        # Check for missing module error
        if "ModuleNotFoundError" in error_msg:
            import re
            match = re.search(r"No module named '([\w_\-]+)'", error_msg)
            if match:
                missing_module = match.group(1)
                logfire.info("Installing missing module",
                             module=missing_module)

                # Try to install the missing module
                install_proc = await asyncio.create_subprocess_exec(
                    "uv", "add", missing_module,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                install_stdout, install_stderr = await install_proc.communicate()

                if install_proc.returncode == 0:
                    logfire.info("Module installed successfully",
                                 module=missing_module)
                    # Retry script after install
                    returncode, stdout, stderr = await run_script()
                    if returncode == 0:
                        # Validation after successful run
                        is_valid = await validate_files_size(temp_dir_path)
                        if not is_valid:
                            raise ValueError(
                                f"[Validation Failed in {sourcename}]: Downloaded files are not > 1KB")
                        return {"sourcename": sourcename, "verified": True}
                    else:
                        # If still fails after installing, raise for agent retry
                        logfire.error("Script failed after module installation",
                                      module=missing_module, error=stderr.decode().strip())
                        raise ModelRetry(
                            f"Script failed after installing {missing_module}. Error: {stderr.decode().strip()}")
                else:
                    logfire.error("Failed to install module",
                                  module=missing_module, error=install_stderr.decode().strip())
                    raise ValueError(
                        f"Failed to install {missing_module}: {install_stderr.decode().strip()}")

        # For other errors (like pandas parsing), trigger agent retry
        logfire.error("Script execution failed", error=error_msg)
        raise ModelRetry(f"Code execution failed: {error_msg}")

    # Simple validation: check if all files are > 1KB
    is_valid = await validate_files_size(temp_dir_path)
    if not is_valid:
        raise ValueError(
            f"[Validation Failed in {sourcename}]: Downloaded files are not > 1KB")

    return {"sourcename": sourcename, "verified": True}

# ---- Utility ----


async def validate_files_size(temp_dir_path: Path) -> bool:
    """Simple validation: check if all files in temp_dir are > 1KB and cleanup Python files"""
    try:
        # First, validate all non-Python files are > 1KB
        for file_path in temp_dir_path.iterdir():
            if file_path.is_file() and not file_path.name.endswith('.py'):
                file_size = file_path.stat().st_size
                if file_size <= 1:  # 1KB = 1024 bytes
                    return False

        # If validation passes, clean up Python script files
        for file_path in temp_dir_path.iterdir():
            if file_path.is_file() and file_path.name.endswith('.py'):
                file_path.unlink()  # Remove the Python script file

        return True
    except Exception:
        return False


async def ensure_temp_dir(deps: AgentDeps) -> str:
    if deps.temp_dir is None:
        deps.temp_dir = str(await create_temp_dir(deps.data_analyst_input))
    return deps.temp_dir

# ---- Orchestrator ----


async def fetch_and_scrape(url: str, temp_dir: str) -> dict:
    """Directly invoke scrape_html without making it a PydanticAI tool."""
    return await scrape_html(url, Path(temp_dir))


async def handle_data(data_analyst_input: str, downloaded_files: List[str]) -> dict:
    with logfire.span("handle_data") as span:
        span.set_attribute("input_length", len(data_analyst_input))
        span.set_attribute("file_count", len(downloaded_files))

        deps = AgentDeps(data_analyst_input=data_analyst_input,
                         downloaded_files=downloaded_files)
        input_hash = hashlib.sha256(data_analyst_input.encode()).hexdigest()

        # Step 1: Fetch or reuse plan
        if input_hash in input_plan_cache:
            plan = input_plan_cache[input_hash]
            logfire.info("Using cached plan", hash=input_hash[:8])
        else:
            with logfire.span("generate_plan"):
                result = await agent.run(data_analyst_input, deps=deps)
                try:
                    plan = DataPlan(root=result.output)
                    logfire.info("Plan generated successfully",
                                 sources=len(result.output))
                except Exception as e:
                    logfire.warn(
                        "Plan generation failed, using fallback", error=str(e))
                    plan = DataPlan(root={
                        "fallback": DataSourcePlan(
                            sourcename="fallback",
                            data_download_pythonic_code=""
                        )
                    })
            input_plan_cache[input_hash] = plan

        # Step 2: Run each source plan using appropriate tool
        results = {}
        for name, ds in plan.root.items():
            with logfire.span("process_source", source=name):
                try:
                    if ds.html and ds.url:
                        # For HTML sources, use the html_handler directly
                        temp_dir = await ensure_temp_dir(deps)
                        result = await fetch_and_scrape(ds.url, temp_dir)
                        results[name] = {"sourcename": name, **result}
                        logfire.info("HTML source processed",
                                     source=name, url=ds.url)
                    else:
                        # For other sources, execute the scraping code and validate file sizes
                        temp_dir_path = Path(await ensure_temp_dir(deps))

                        # Replace {tmp} placeholder with actual temp directory path
                        scraping_code = ds.data_download_pythonic_code.replace(
                            "{tmp}", str(temp_dir_path))

                        # Write scrape script
                        script_path = temp_dir_path / \
                            f"{ds.sourcename}_scrape.py"
                        async with aiofiles.open(script_path, "w") as f:
                            await f.write(scraping_code)

                        # Execute scrape script
                        proc = await asyncio.create_subprocess_exec(
                            "python", str(script_path),
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE
                        )
                        stdout, stderr = await proc.communicate()
                        if proc.returncode != 0:
                            raise ValueError(
                                f"[Scraping Error in {ds.sourcename}]: {stderr.decode().strip()}")

                        # Simple validation: check if all files are > 1KB
                        is_valid = await validate_files_size(temp_dir_path)
                        if not is_valid:
                            raise ValueError(
                                f"[Validation Failed in {ds.sourcename}]: Downloaded files are not > 1KB")

                        results[name] = {
                            "sourcename": ds.sourcename, "verified": True}
                        logfire.info("Data source processed",
                                     source=name, verified=True)

                except Exception as e:
                    results[name] = {"error": str(e)}
                    logfire.error("Source processing failed",
                                  source=name, error=str(e))

        # Return results along with temp directory path
        temp_dir = await ensure_temp_dir(deps)

        logfire.info("Data processing completed",
                     total_sources=len(plan.root),
                     successful_sources=len([r for r in results.values() if "error" not in r]))

        result = {
            "results": results,
            "temp_directory": temp_dir,
            "total_sources": len(plan.root),
            "successful_sources": len([r for r in results.values() if "error" not in r])
        }

        # Convert any numpy types to native Python types
        return convert_numpy_types(result)
