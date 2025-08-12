# file: llm_code_runner.py

import os  # For environment variable access
import sys  # For getting the current Python executable path
import asyncio  # For running async functions
import subprocess  # For executing generated code
from pathlib import Path  # For filesystem path management
from typing import List, Dict  # For type hints
# For LLM structured output handling
from pydantic_ai import Agent, StructuredDict
# For Google Gemini model integration
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider  # For Google API provider
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
# Schema that enforces the LLM output structure
code_schema = {
    "type": "object",
    "properties": {
        "data_analysis_pythonic_code": {
            "type": "string",
            "description": (
                "Python code that answers the questions in the data analyst input "
                "and outputs strictly in the requested format."
            )
        }
    },
    "required": ["data_analysis_pythonic_code"]
}

# --- Config ---
GEMINI_KEY = os.getenv("GEMINI_API_KEY")  # Load API key from environment
if not GEMINI_KEY:
    raise ValueError("GEMINI_API_KEY not set")  # Fail early if key missing

# Configure Google provider and model
provider = GoogleProvider(api_key=GEMINI_KEY)
model = GoogleModel("gemini-2.5-flash", provider=provider)

# Create the LLM agent with structured output
agent = Agent(
    model,
    output_type=StructuredDict(code_schema),
    # tools=[duckduckgo_search_tool]  # Add search tool for external queries
)

# Conversation memory to keep track of prompt/response iterations
memory: List[Dict[str, str]] = []

# Main async function to run data analysis attempts


async def answer_questions(data_analyst_input: str, temp_dir: str, max_attempts: int = 3):
    # Read analysis report content from file if it exists
    analysis_report_path = Path(temp_dir) / "analysis_report.txt"
    analysis_report_content = ""
    if analysis_report_path.exists():
        analysis_report_content = analysis_report_path.read_text()

    # Attempt loop for retrying failed executions
    for attempt in range(1, max_attempts + 1):
        print(f"\n--- Attempt {attempt} ---")

        # Combine memory history into conversation context
        conversation_context = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in memory
        )

        # Build prompt for this attempt
        prompt = f"""
You are a precise Python data analyst.

INPUT:
DATA ANALYST INPUT (task + required output format):
{data_analyst_input}

DATA STRUCTURE ANALYSIS:
{analysis_report_content}

TEMP DIR:
{temp_dir}

TASK:
- use search tool if needed
- Write valid Python code that:
  1) Answers all questions in DATA ANALYST INPUT using the data.
  2) Outputs strictly in the requested format (convert to valid JSON).
  3) Saves any intermediate files in TEMP DIR.
  4) Includes all imports.
  5) At the end, the code should convert all outputs to JSON serializable and requested format.
  6) If HTML analysis present give code that analyses HTML which is in the {temp_dir} and answers questions based on it.
- No extra text; only the code as 'data_analysis_pythonic_code'.
"""

        # Append user prompt to memory
        memory.append({"role": "user", "content": prompt.strip()})

        # Query the LLM for code
        result = await agent.run(prompt)

        # Extract the Python code string from structured output
        code = result.output["data_analysis_pythonic_code"]

        # Append LLM response to memory
        memory.append({"role": "assistant", "content": code})

        # Save generated code to a file for execution
        code_path = Path(temp_dir) / f"generated_code_attempt_{attempt}.py"
        code_path.write_text(code)

        # Execute generated code using the same Python interpreter
        process = subprocess.run(
            [sys.executable, str(code_path)],
            capture_output=True,
            text=True
        )

        # Check execution success
        if process.returncode == 0:
            print("\n✅ Execution Successful")
            print(process.stdout)
            return process.stdout
        else:
            print("\n❌ Execution Failed")
            print(process.stderr)

            # Add error message to memory for LLM self-correction
            memory.append({
                "role": "user",
                "content": f"Execution error:\n{process.stderr}\nPlease fix the code."
            })

    # Raise if all attempts fail
    raise RuntimeError(
        "Max attempts reached. Code did not execute successfully.")
