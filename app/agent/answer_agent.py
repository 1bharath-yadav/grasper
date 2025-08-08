from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import os
import asyncio
import aiofiles
import time

from dotenv import load_dotenv
from pydantic import BaseModel, RootModel
from pydantic_ai import Agent, RunContext, StructuredDict
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider
import logfire

from .utils import convert_numpy_types

load_dotenv()

# Configure Logfire
logfire.configure()

# ---- Agent Dependency Types ----


@dataclass
class AnswerDeps:
    data_analyst_input: str
    temp_dir: str
    analysis_report: str
    file_analysis: Dict[str, Any]


@dataclass
class FormatDeps:
    questions: List[str]
    raw_answers: str
    original_request: str

# ---- Schema Definitions ----


class QuestionAnalysis(BaseModel):
    questions: List[str]
    data_analysis_pythonic_code: str


question_analysis_schema = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {"type": "string"}
        },
        "data_analysis_pythonic_code": {"type": "string"}
    },
    "required": ["questions", "data_analysis_pythonic_code"]
}

# ---- LLM Model Setup ----


gemini_key = os.getenv("GEMINI_API_KEY")
if not gemini_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

settings = GoogleModelSettings(google_thinking_config={"thinking_budget": 0})
provider = GoogleProvider(api_key=gemini_key)
model = GoogleModel("gemini-2.5-flash", provider=provider)

# ---- Agent Initialization ----

# First agent: Extract questions and generate code
question_code_agent = Agent(
    model,
    output_type=StructuredDict(question_analysis_schema),
    deps_type=AnswerDeps,
    system_prompt=(
        "You are a data analyst question extractor and code generator. "
        "Extract all questions from the data_analyst_input and generate Python code to answer them. "
        "Your response should have:\n"
        "1. 'questions': Array of individual questions extracted from the input\n"
        "2. 'data_analysis_pythonic_code': Clean Python code (no markdown formatting) that reads files from temp directory and answers all questions\n\n"
        "IMPORTANT PLOTTING GUIDELINES:\n"
        "- If a question asks for a plot/chart/graph, create the visualization using matplotlib\n"
        "- If the question specifically requests 'base64' or 'encode as base64', convert the plot to base64 PNG format\n"
        "- For base64 plots, use this pattern:\n"
        "  import matplotlib.pyplot as plt\n"
        "  import base64\n"
        "  from io import BytesIO\n"
        "  # ... create your plot ...\n"
        "  buf = BytesIO()\n"
        "  plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)\n"
        "  buf.seek(0)\n"
        "  plot_base64 = base64.b64encode(buf.read()).decode('utf-8')\n"
        "  plt.close()\n"
        "  print(f'Base64 PNG: {plot_base64}')\n\n"
        "- Always use plt.close() after saving plots to free memory\n"
        "- Convert all numpy types to native Python types when printing results\n"
        "- The code should print raw results only, no explanatory text"
    ),
    retries=2,
)

# Second agent: Format the final response
format_agent = Agent(
    model,
    deps_type=FormatDeps,
    system_prompt=(
        "You are a response formatter. Given questions and raw answers, format them according to the original request. "
        "Look at the original request to understand the exact format needed (JSON array, JSON object, etc.) "
        "and format the raw answers accordingly. Return ONLY the final formatted result."
    ),
    retries=2,
)

# ---- Analysis Functions ----


async def read_analysis_report(temp_dir: str) -> str:
    """Read the analysis report from temp directory."""
    report_path = Path(temp_dir) / "analysis_report.txt"

    if not report_path.exists():
        logfire.warn("Analysis report not found", path=str(report_path))
        return "No analysis report available."

    try:
        async with aiofiles.open(report_path, 'r', encoding='utf-8') as f:
            return await f.read()
    except Exception as e:
        logfire.error("Failed to read analysis report", error=str(e))
        return f"Error reading analysis report: {str(e)}"


async def execute_analysis_code(code: str, temp_dir: str) -> Dict[str, Any]:
    """Execute analysis code and return results."""
    script_path = Path(temp_dir) / "answer_analysis.py"

    try:
        # Write analysis script
        async with aiofiles.open(script_path, "w") as f:
            await f.write(code)

        # Execute script
        proc = await asyncio.create_subprocess_exec(
            "python", str(script_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=temp_dir
        )
        stdout, stderr = await proc.communicate()

        # Clean up script (comment out for debugging)
        # try:
        #     script_path.unlink()
        # except OSError:
        #     pass

        if proc.returncode != 0:
            error_msg = stderr.decode().strip()
            logfire.error("Analysis code execution failed",
                          error=error_msg,
                          script_path=str(script_path))
            return {
                "error": f"Execution failed: {error_msg}",
                "success": False,
                # Include script path for debugging
                "script_path": str(script_path)
            }

        # Return the stdout output (actual results)
        output = stdout.decode().strip()
        logfire.info("Analysis code executed successfully",
                     output_length=len(output))

        return {"output": output, "success": True}

    except Exception as e:
        logfire.error("Failed to execute analysis code", error=str(e))
        return {"error": str(e), "success": False}

# ---- Main Answer Function ----


async def generate_answers(
    data_analyst_input: str,
    temp_dir: str,
    file_analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate answers to questions based on analyzed data using two-step process."""

    start_time = time.time()

    with logfire.span("generate_answers") as span:
        span.set_attribute("temp_dir", temp_dir)
        span.set_attribute("input_length", len(data_analyst_input))

        # Read analysis report
        analysis_report = await read_analysis_report(temp_dir)

        # Create the complete prompt with data_analyst_input and analysis_report
        complete_prompt = f"""
DATA ANALYST INPUT:
{data_analyst_input}

ANALYSIS REPORT:
{analysis_report}

TEMP DIRECTORY PATH: {temp_dir}

Extract all questions from the data_analyst_input and generate Python code to answer them using the data files in the temp directory.
"""

        deps = AnswerDeps(
            data_analyst_input=data_analyst_input,
            temp_dir=temp_dir,
            analysis_report=analysis_report,
            file_analysis=file_analysis
        )

        try:
            # STEP 1: Extract questions and generate code
            with logfire.span("step1_extract_and_generate"):
                step1_result = await question_code_agent.run(complete_prompt, deps=deps)
                questions = step1_result.output["questions"]
                generated_code = step1_result.output["data_analysis_pythonic_code"]

                logfire.info("Step 1 completed",
                             questions_count=len(questions),
                             code_length=len(generated_code))

            # STEP 2: Execute the generated code to get raw answers
            with logfire.span("step2_execute_code"):
                execution_result = await execute_analysis_code(generated_code, temp_dir)

                if not execution_result["success"]:
                    logfire.error("Code execution failed",
                                  error=execution_result["error"])
                    result = {
                        "status": "error",
                        "answer": f"Code execution failed: {execution_result['error']}",
                        # Convert to strings
                        "questions": [str(q) for q in questions],
                        # Convert to string
                        "generated_code": str(generated_code),
                        "temp_directory": str(temp_dir),
                        # Convert to float
                        "execution_time": float(time.time() - start_time),
                        "error": str(execution_result["error"])
                    }
                    # Convert any numpy types to native Python types
                    return convert_numpy_types(result)

                raw_answers = execution_result["output"]
                logfire.info("Step 2 completed",
                             raw_answers_length=len(raw_answers))

            # STEP 3: Format the final response using the second agent
            with logfire.span("step3_format_response"):
                format_deps = FormatDeps(
                    questions=questions,
                    raw_answers=raw_answers,
                    original_request=data_analyst_input
                )

                format_prompt = f"""
ORIGINAL REQUEST:
{data_analyst_input}

QUESTIONS EXTRACTED:
{json.dumps(questions, indent=2)}

RAW ANSWERS FROM CODE EXECUTION:
{raw_answers}

Format the raw answers according to the exact format requested in the original request.
"""

                try:
                    # Add timeout to prevent hanging
                    format_result = await asyncio.wait_for(
                        format_agent.run(format_prompt, deps=format_deps),
                        timeout=30.0  # 30 second timeout
                    )
                    final_formatted_answer = format_result.output

                    logfire.info("Step 3 completed", formatted_answer_length=len(
                        final_formatted_answer))
                except asyncio.TimeoutError:
                    logfire.warn(
                        "Format agent timed out, returning raw answers")
                    final_formatted_answer = raw_answers
                except Exception as e:
                    logfire.error(
                        "Format agent failed, returning raw answers", error=str(e))
                    final_formatted_answer = raw_answers

            execution_time = time.time() - start_time

            result = {
                "status": "success",
                "answer": final_formatted_answer,  # Final formatted answer
                "questions": [str(q) for q in questions],  # Convert to strings
                "raw_answers": str(raw_answers),  # Convert to string
                "generated_code": str(generated_code),  # Convert to string
                "temp_directory": str(temp_dir),
                "execution_time": float(execution_time)  # Convert to float
            }

            # Convert any numpy types to native Python types
            return convert_numpy_types(result)

        except Exception as e:
            execution_time = time.time() - start_time
            logfire.error("Failed to generate answers", error=str(e))

            result = {
                "status": "error",
                "answer": f"Analysis failed: {str(e)}",
                "temp_directory": temp_dir,
                "execution_time": execution_time,
                "error": str(e)
            }

            # Convert any numpy types to native Python types
            return convert_numpy_types(result)

# ---- Public Interface ----


async def answer_questions(
    data_analyst_input: str,
    temp_dir: str,
    file_analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """Main function to answer questions based on analyzed data."""
    return await generate_answers(data_analyst_input, temp_dir, file_analysis)
