"""
Orchestrator Agent using Pydantic AI for coordinating data analysis workflows.
"""

import json
import re
import asyncio
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from app.agent.utils import convert_numpy_types
import logfire
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext, StructuredDict

from app.agent.files_download_agent import handle_urls
from app.agent.file_analysis_handler import analyze_all_files
from app.agent.answer_agent import answer_questions
from app.agent.html_handler import analyze_html_async
from app.config import get_gemini_model, get_gemini_model


@dataclass
class OrchestrationContext:
    """Context passed to the orchestrator agent."""
    uploaded_files_paths: List[Path]
    data_analysis_input: str
    temp_dir: str


# will be implement NOTE
OrchestrationResultSchema = {
    "title": "OrchestrationResult",
    "type": "object",
    "description": "Structured output from orchestration process",
    "properties": {
        "status": {
            "type": "string",
            "description": "Overall status of the orchestration process",
            "enum": ["success", "partial_success", "error", "validation_failed"]
        },
        "result": {
            "type": "object",
            "description": "Main result data from the orchestration",
            "properties": {
                "final_answer": {
                    "description": "User's requested format response"
                },
                "output": {
                    "description": "Alternative output field if final_answer not present"
                }
            },
            "additionalProperties": True
        },
        "analysis_report": {
            "type": "object",
            "description": "File analysis results",
            "properties": {
                "stats": {
                    "type": "object",
                    "properties": {
                        "n_files": {
                            "type": "integer",
                            "description": "Number of files analyzed"
                        }
                    },
                    "additionalProperties": True
                }
            },
            "additionalProperties": True
        },
        "html_results": {
            "type": "object",
            "description": "HTML analysis results from web scraping",
            "additionalProperties": True
        },
        "downloaded_files": {
            "type": "array",
            "description": "List of files downloaded from URLs",
            "items": {
                "type": "string",
                "description": "File path of downloaded file"
            }
        },
        "error": {
            "type": "string",
            "description": "Error message if orchestration failed"
        },
        "workflow_steps": {
            "type": "array",
            "description": "List of workflow steps executed",
            "items": {
                "type": "string",
                "description": "Description of workflow step (e.g., 'File Download', 'HTML Analysis', etc.)"
            }
        },
        "final_user_response": {
            "description": "The final response in the exact format requested by user - can be any type",
            "type": "string"
        }
    },
    "required": ["status"],
    "additionalProperties": False
}


# Initialize the orchestrator agent
orchestrator_agent = Agent(
    model=get_gemini_model(),
    output_type=StructuredDict(OrchestrationResultSchema),
    deps_type=OrchestrationContext,
    system_prompt="""You are an intelligent orchestrator agent that coordinates data analysis workflows.
    
    Your job is to:
    1. Analyze user input to determine which processing agents to invoke
    2. Coordinate the execution of file download, HTML scraping, and file analysis
    3. Ensure all results are properly integrated and returned in the EXACT format requested by the user
    
    Decision Rules:
    - If input contains file download URLs → invoke Files Download Agent
    - If input contains website URLs & HTML-related questions → invoke HTML Handler Agent
    - If both conditions are true → invoke agents sequentially and merge outputs
    - Always ensure file analysis is performed on available files
    - CRITICAL: Return the final answer in exactly the format the user requested (JSON, table, summary, etc.)
    
    IMPORTANT OUTPUT INSTRUCTIONS:
    - When generate_answers_tool returns a result with "final_answer", put that content in the "final_user_response" field
    - The "final_user_response" should contain the user's requested answer format directly
    - Keep the structure simple and focused on delivering the user's requested output
    
    Always ensure proper error handling and provide meaningful status updates.
    Return comprehensive results that integrate all analysis outputs and present them in the user's requested format."""
)


@orchestrator_agent.tool
async def download_files_tool(ctx: RunContext[OrchestrationContext]) -> Dict[str, Any]:
    """Download files from URLs found in the input."""
    logfire.info("Starting file download tool")
    try:
        logfire.info(f"Found URLs to download")
        downloaded_files = await handle_urls(ctx.deps.data_analysis_input, ctx.deps.temp_dir)
        logfire.info(f"Successfully downloaded {len(downloaded_files)} files")
        return {
            "status": "success",
            "files": [str(f) for f in downloaded_files],
            "count": len(downloaded_files)
        }
    except Exception as e:
        logfire.error("Files download failed", error=str(e))
        return {"status": "error", "error": str(e)}


@orchestrator_agent.tool
async def analyze_html_tool(ctx: RunContext[OrchestrationContext]) -> Dict[str, Any]:
    """Analyze HTML content from websites."""
    logfire.info("Starting HTML analysis tool")
    try:
        # ensure URLs from input
        urls = re.findall(r'https?://[^\s]+', ctx.deps.data_analysis_input)
        if not urls:
            logfire.info("No URLs found for HTML analysis")
            return {"status": "skip", "reason": "No URLs found in input"}

        # Check if HTML-related questions are present
        html_keywords = ['html', 'website', 'web',
                         'scrape', 'crawl', 'page', 'content', 'extract']
        has_html_question = any(
            keyword.lower() in ctx.deps.data_analysis_input.lower() for keyword in html_keywords)

        if not has_html_question:
            logfire.info("No HTML-related questions detected")
            return {"status": "skip", "reason": "No HTML-related questions detected"}

        # Analyze first URL (can be extended to handle multiple URLs)
        logfire.info(f"Analyzing HTML content for {len(urls)} URLs")
        html_results = await analyze_html_async(urls, ctx.deps.data_analysis_input, ctx.deps.temp_dir)
        logfire.info(f"html results:{html_results}")
        logfire.info("HTML analysis completed successfully")
        return {"status": "success", "results": html_results}
    except Exception as e:
        logfire.error("HTML analysis failed", error=str(e))
        return {"status": "error", "error": str(e)}


@orchestrator_agent.tool
async def analyze_files_tool(ctx: RunContext[OrchestrationContext]) -> Dict[str, Any]:
    """Analyze files in the temporary directory."""
    logfire.info("Starting file analysis tool")
    try:
        logfire.info(f"Analyzing files in temp directory: {ctx.deps.temp_dir}")
        print(
            f"[DEBUG] Analyzing files in temp directory: {ctx.deps.temp_dir}")
        analysis_report = await analyze_all_files(ctx.deps.temp_dir, ctx.deps.data_analysis_input)
        logfire.info(
            f"File analysis completed. Analyzed {analysis_report.get('stats', {}).get('n_files', 0)} files")
        return {"status": "success", "report": analysis_report}
    except Exception as e:
        logfire.error("File analysis failed", error=str(e))
        raise ModelRetry(f"File analysis failed: {str(e)}")


@orchestrator_agent.tool
async def generate_answers_tool(ctx: RunContext[OrchestrationContext], analysis_data: Optional[Dict[str, Any]] = None, html_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Generate final answers based on analysis."""
    logfire.info("Starting answer generation tool")
    try:
        # Prepare context for answer generation
        context = {
            "analysis_report": analysis_data,
            "html_results": html_data,
            "original_query": ctx.deps.data_analysis_input
        }

        logfire.info("Calling answer_questions function")
        final_result = await answer_questions(
            ctx.deps.data_analysis_input,
            ctx.deps.temp_dir,
            max_attempts=3,
            context=context
        )
        logfire.info("Answer generation completed successfully")
        logfire.info(f"Final result type: {type(final_result)}")
        logfire.info(
            f"Final result keys: {list(final_result.keys()) if isinstance(final_result, dict) else 'Not a dict'}")

        # Simplify the return to make it easier for orchestrator agent to process
        # Extract just the final_answer if it exists
        if isinstance(final_result, dict) and "final_answer" in final_result:
            simplified_result = final_result["final_answer"]
            logfire.info("Extracted final_answer for orchestrator")
        else:
            simplified_result = final_result
            logfire.info("Using full result as final_answer")

        tool_return = {"status": "success", "final_answer": simplified_result}
        logfire.info(f"Tool returning simplified: {tool_return}")
        return tool_return
    except Exception as e:
        logfire.error("Answer generation failed", error=str(e))
        raise ModelRetry(f"Answer generation failed: {str(e)}")


async def orchestrator(
    uploaded_files_paths: List[Path],
    data_analysis_input: str,
    temp_dir: str
) -> Any:
    """
    Main orchestrator function - using direct tool execution for reliability.

    Args:
        uploaded_files_paths: List of paths to user-uploaded files
        data_analysis_input: User's questions and data analysis requirements
        temp_dir: Temporary directory path for processing

    Returns:
        Structured analysis results
    """
    with logfire.span("orchestrator_direct") as span:
        span.set_attribute("temp_dir", temp_dir)
        span.set_attribute("uploaded_files_paths_count",
                           len(uploaded_files_paths))

        context = OrchestrationContext(
            uploaded_files_paths=uploaded_files_paths,
            data_analysis_input=data_analysis_input,
            temp_dir=temp_dir
        )

        try:
            logfire.info(
                "Starting direct orchestration - calling functions directly")

            # Step 1: Check for URLs and download files if present
            urls = re.findall(r'https?://[^\s]+', data_analysis_input)
            downloaded_files = []
            if urls:
                logfire.info(f"Found {len(urls)} URLs, downloading files")
                try:
                    downloaded_files = await handle_urls(data_analysis_input, temp_dir)
                    logfire.info(
                        f"Successfully downloaded {len(downloaded_files)} files")
                except Exception as e:
                    logfire.error(f"File download failed: {str(e)}")

            # Step 2: Check for HTML analysis needs
            html_results = None
            html_keywords = ['html', 'website', 'web', 'scrape',
                             'crawl', 'page', 'content', 'extract', 'wikipedia']
            has_html_question = any(
                keyword.lower() in data_analysis_input.lower() for keyword in html_keywords)

            if urls and has_html_question:
                logfire.info(
                    f"HTML-related keywords detected, analyzing {len(urls)} URLs")
                try:
                    html_results = await analyze_html_async(urls, data_analysis_input, temp_dir)
                    # Check if HTML analysis actually succeeded
                    if isinstance(html_results, dict) and html_results.get("status") == "error":
                        logfire.error(
                            f"HTML analysis failed: {html_results.get('user_requested_response', 'Unknown error')}")
                        html_results = None
                    else:
                        logfire.info("HTML analysis completed successfully")
                except Exception as e:
                    logfire.error(f"HTML analysis failed: {str(e)}")
                    html_results = None

            # Step 3: Analyze files directly
            logfire.info("Running file analysis directly")
            analysis_report = await analyze_all_files(temp_dir, data_analysis_input)
            logfire.info(
                f"File analysis completed. Analyzed {analysis_report.get('stats', {}).get('n_files', 0)} files")

            # Step 4: Generate answers directly with all available context
            logfire.info("Running answer generation directly")

            context = {
                "analysis_report": analysis_report,
                "html_results": html_results,
                "original_query": data_analysis_input,
                "downloaded_files": downloaded_files
            }
            logfire.info(f"Context prepared for answer generation: {context}")
            print(f"[DEBUG] Context prepared for answer generation: {context}")

            final_result = await answer_questions(
                data_analysis_input,
                temp_dir,
                max_attempts=3,
                context=context
            )

            # Extract the final answer
            if isinstance(final_result, dict) and "final_answer" in final_result:
                final_answer = final_result["final_answer"]
                logfire.info("Direct orchestration completed successfully")
                return final_answer
            else:
                logfire.info("Returning full result from answer generation")
                return json.dumps(convert_numpy_types(final_result))

        except Exception as e:
            logfire.error("Direct orchestration failed", error=str(e),
                          traceback=traceback.format_exc())
            return {
                "status": "error",
                "error": f"Direct orchestration failed: {str(e)}",
                "workflow_steps": ["Failed during direct orchestration"]
            }


# Alternative direct orchestration function for simpler use cases
async def orchestrate_direct(
    uploaded_files_paths: List[Path],
    data_analysis_input: str,
    temp_dir: str
) -> Dict[str, Any]:
    """
    Direct orchestration without Pydantic AI agent (for simpler workflows).

    Args:
        uploaded_files_paths: List of paths to user-uploaded files
        data_analysis_input: User's questions and data analysis requirements
        temp_dir: Temporary directory path for processing

    Returns:
        Structured analysis results
    """
    logfire.info("Starting direct orchestration")

    result = {
        "status": "success",
        "workflow_steps": [],
        "downloaded_files": [],
        "analysis_report": None,
        "html_results": None,
        "final_result": None,
        "errors": []
    }

    try:
        # Step 1: Download files if URLs are present
        urls = re.findall(r'https?://[^\s]+', data_analysis_input)
        if urls:
            logfire.info("Downloading files from URLs")
            try:
                downloaded_files = await handle_urls(data_analysis_input, temp_dir)
                result["downloaded_files"] = [str(f) for f in downloaded_files]
                result["workflow_steps"].append("File Download - Success")
                logfire.info(f"Downloaded {len(downloaded_files)} files")
            except Exception as e:
                error_msg = f"File download failed: {str(e)}"
                result["errors"].append(error_msg)
                result["workflow_steps"].append("File Download - Failed")
                logfire.error(error_msg)

        # Step 2: Analyze HTML if needed
        html_keywords = ['html', 'website', 'web',
                         'scrape', 'crawl', 'page', 'content', 'extract']
        has_html_question = any(
            keyword.lower() in data_analysis_input.lower() for keyword in html_keywords)

        if urls and has_html_question:
            logfire.info("Analyzing HTML content")
            try:
                html_results = await analyze_html_async(urls, data_analysis_input, temp_dir)
                result["html_results"] = html_results
                result["workflow_steps"].append("HTML Analysis - Success")
                logfire.info("HTML analysis completed")
            except Exception as e:
                error_msg = f"HTML analysis failed: {str(e)}"
                result["errors"].append(error_msg)
                result["workflow_steps"].append("HTML Analysis - Failed")
                logfire.error(error_msg)

        # Step 3: Analyze all files
        logfire.info("Analyzing files")
        try:
            analysis_report = await analyze_all_files(temp_dir, data_analysis_input)
            result["analysis_report"] = analysis_report
            result["workflow_steps"].append("File Analysis - Success")
            logfire.info("File analysis completed")
        except Exception as e:
            error_msg = f"File analysis failed: {str(e)}"
            result["errors"].append(error_msg)
            result["workflow_steps"].append("File Analysis - Failed")
            logfire.error(error_msg)

        # Step 4: Generate final answers
        logfire.info("Generating final answers")
        try:
            context = {
                "analysis_report": result.get("analysis_report"),
                "html_results": result.get("html_results"),
                "original_query": data_analysis_input
            }

            final_result = await answer_questions(
                data_analysis_input,
                temp_dir,
                max_attempts=3,
                context=context
            )
            result["final_result"] = final_result
            result["workflow_steps"].append("Answer Generation - Success")
            logfire.info("Answer generation completed")
        except Exception as e:
            error_msg = f"Answer generation failed: {str(e)}"
            result["errors"].append(error_msg)
            result["workflow_steps"].append("Answer Generation - Failed")
            logfire.error(error_msg)

        # Set final status
        if result["errors"]:
            result["status"] = "partial_success" if any(
                "Success" in step for step in result["workflow_steps"]) else "error"
        else:
            result["status"] = "success"

        return result

    except Exception as e:
        logfire.error("Critical orchestration failure", error=str(e))
        return {
            "status": "error",
            "error": f"Critical orchestration failure: {str(e)}",
            "workflow_steps": ["Orchestration Failed"],
            "errors": [str(e)]
        }
