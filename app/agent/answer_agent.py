# file: answer_agent.py

import os
import sys
import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext

from app.config import get_gemini_model, get_gemini_model


@dataclass
class AnswerContext:
    """Context for answer generation."""
    data_analysis_input: str
    temp_dir: str
    analysis_report: Optional[Dict[str, Any]] = None
    html_results: Optional[Dict[str, Any]] = None
    downloaded_files: Optional[List[str]] = None
    max_attempts: int = 3


class CodeExecutionResult(BaseModel):
    """Result from code execution."""
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    code_path: Optional[str] = None


class GeneratedCode(BaseModel):
    """Schema for LLM code generation output."""
    data_analysis_pythonic_code: str = Field(
        description="Complete Python code that answers the analysis questions and outputs in requested format"
    )
    explanation: Optional[str] = Field(
        default=None,
        description="Brief explanation of the approach taken"
    )


# Initialize the answer generation agent
answer_agent = Agent(
    model=get_gemini_model(),
    deps_type=AnswerContext,
    output_type=GeneratedCode,
    system_prompt="""You are a precise Python data analyst that generates executable code that 

1. Analyze the provided data analysis requirements
2. Generate complete, executable Python code
3. Ensure output matches the exact format requested by the user
4. Handle various data sources (files, HTML, APIs)
5. Include proper error handling and validation

Code Requirements:
- Include ALL necessary imports at the top
- Use the provided temp directory for file operations
- Convert final outputs to JSON serializable format
- Output in the EXACT format requested by the user

Never generate incomplete code or placeholder functions."""
)


@answer_agent.tool
async def validate_code_syntax(ctx: RunContext[AnswerContext], code: str) -> Dict[str, Any]:
    """Validate Python code syntax before execution."""
    try:
        compile(code, '<string>', 'exec')
        return {"valid": True, "message": "Code syntax is valid"}
    except SyntaxError as e:
        return {
            "valid": False,
            "error": str(e),
            "line": e.lineno,
            "message": f"Syntax error at line {e.lineno}: {e.msg}"
        }


async def execute_code(code: str, temp_dir: str, attempt: int) -> CodeExecutionResult:
    """Execute generated Python code safely."""
    code_path = Path(temp_dir) / f"generated_code_attempt_{attempt}.py"

    try:
        # Write code to file
        code_path.write_text(code, encoding='utf-8')
        logfire.info(f"Code written to {code_path}")

        # Execute code with timeout using uv run to ensure proper environment
        # Find project root (where pyproject.toml is located)
        current_dir = Path(temp_dir)
        project_root = None

        # Look for project root by searching up the directory tree
        search_path = current_dir
        while search_path.parent != search_path:  # Stop at filesystem root
            if (search_path / "pyproject.toml").exists():
                project_root = str(search_path)
                break
            search_path = search_path.parent

        # If no pyproject.toml found, try current working directory
        if not project_root:
            cwd = Path.cwd()
            if (cwd / "pyproject.toml").exists():
                project_root = str(cwd)
            else:
                project_root = temp_dir  # Fallback to temp_dir

        try:
            # Try to use uv run for better environment isolation
            process = await asyncio.create_subprocess_exec(
                "uv", "run", "python", str(code_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=project_root
            )
        except FileNotFoundError:
            # Fall back to direct python execution if uv is not available
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(code_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=temp_dir
            )

        try:
            # 5 min timeout
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)
            logfire.debug(f"Code execution stdout: {stdout.decode('utf-8')}")
            logfire.debug(f"Code execution stderr: {stderr.decode('utf-8')}")
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return CodeExecutionResult(
                success=False,
                error="Code execution timed out after 5 minutes",
                code_path=str(code_path)
            )

        stdout_text = stdout.decode('utf-8')
        stderr_text = stderr.decode('utf-8')

        # Consider execution successful if return code is 0, even if there's stderr output
        # (many libraries produce warnings on stderr that don't indicate failure)
        if process.returncode == 0:
            logfire.info(f"Code execution successful (attempt {attempt})")
            return CodeExecutionResult(
                success=True,
                output=stdout_text,
                code_path=str(code_path)
            )
        else:
            # Log detailed error information
            logfire.error(
                f"Code execution failed (attempt {attempt})",
                error=stderr_text,
                return_code=process.returncode,
                stdout=stdout_text[:500] if stdout_text else "No stdout"
            )
            return CodeExecutionResult(
                success=False,
                error=f"Exit code {process.returncode}: {stderr_text}" if stderr_text else f"Exit code {process.returncode} with no error message",
                code_path=str(code_path)
            )

    except Exception as e:
        logfire.error(
            f"Code execution exception (attempt {attempt})", error=str(e))
        return CodeExecutionResult(
            success=False,
            error=f"Execution exception: {str(e)}",
            code_path=str(code_path) if code_path.exists() else None
        )


def build_analysis_context(analysis_report: Optional[Dict[str, Any]],
                           html_results: Optional[Dict[str, Any]],
                           temp_dir: str) -> str:
    """Build comprehensive context string for code generation."""
    context_parts = []

    # Add file analysis context
    if analysis_report:
        context_parts.append("=== FILE ANALYSIS REPORT ===")
        context_parts.append(json.dumps(analysis_report, indent=2))

        # Add file paths information
        if analysis_report.get('file_summaries'):
            context_parts.append("\n=== AVAILABLE FILES ===")
            for file_info in analysis_report['file_summaries']:
                context_parts.append(
                    f"- {file_info.get('file_path', 'Unknown path')}: {file_info.get('file_type', 'Unknown type')}")

    # Add HTML analysis context
    if html_results:
        context_parts.append("\n=== HTML ANALYSIS RESULTS ===")
        context_parts.append(json.dumps(html_results, indent=2))

    # Add temp directory info
    context_parts.append(f"\n=== WORKING DIRECTORY ===")
    context_parts.append(f"Temp directory: {temp_dir}")

    # List available files in temp directory
    temp_path = Path(temp_dir)
    if temp_path.exists():
        files = list(temp_path.glob("*"))
        if files:
            context_parts.append("Available files:")
            for file in files:
                context_parts.append(f"- {file.name} ({file.suffix})")
        else:
            context_parts.append("No files currently in temp directory")

    return "\n".join(context_parts)


async def answer_questions(
    data_analysis_input: str,
    temp_dir: str,
    max_attempts: int = 3,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Main answer generation function using Pydantic AI agent.

    Args:
        data_analysis_input: User's analysis questions and requirements
        temp_dir: Temporary directory for file operations
        max_attempts: Maximum number of code generation attempts
        context: Additional context from orchestrator (analysis_report, html_results)

    Returns:
        Dictionary containing the final results and execution details
    """
    with logfire.span("answer_generation") as span:
        span.set_attribute("temp_dir", temp_dir)
        span.set_attribute("max_attempts", max_attempts)
        span.set_attribute("input_length", len(data_analysis_input))

        # Ensure temp directory exists
        Path(temp_dir).mkdir(parents=True, exist_ok=True)

        # Extract context data
        analysis_report = context.get('analysis_report') if context else None
        html_results = context.get('html_results') if context else None

        # Build comprehensive analysis context
        analysis_context = build_analysis_context(
            analysis_report, html_results, temp_dir)

        # Initialize memory for conversation tracking
        memory = []
        execution_results = []

        for attempt in range(1, max_attempts + 1):
            logfire.info(f"Starting answer generation attempt {attempt}")

            # Build conversation context from memory
            conversation_context = "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in memory
            ) if memory else ""

            # Construct the prompt for code generation
            if attempt == 1:
                prompt = f"""
Generate Python code to answer the following data analysis request:

USER REQUEST:
{data_analysis_input}

AVAILABLE DATA CONTEXT:
{analysis_context}

CONVERSATION HISTORY:
{conversation_context}

REQUIREMENTS:
1. Write complete, executable Python code
3. Include ALL necessary imports
4. Answer ALL questions in the user request
5. Output in the EXACT format requested by the user
6. Handle file reading/writing properly
7. If HTML data exists, read HTML files from temp_dir and analyze them
8. Print the final results clearly
"""
            else:
                # For retry attempts, include the previous error
                last_error = execution_results[-1].error if execution_results else "Unknown error"
                prompt = f"""
The previous code failed with error: {last_error}

Please fix the code and generate a corrected version that:
1. Addresses the specific error that occurred
2. Maintains all original requirements
3. Is more robust and handles edge cases
4. Still answers the original user request: {data_analysis_input}

AVAILABLE DATA CONTEXT:
{analysis_context}

CONVERSATION HISTORY:
{conversation_context}

Generate the corrected Python code.
"""

            # Add prompt to memory
            memory.append({"role": "user", "content": prompt})

            try:
                # Generate code using Pydantic AI agent
                result = await answer_agent.run(prompt, deps=AnswerContext(
                    data_analysis_input=data_analysis_input,
                    temp_dir=temp_dir,
                    analysis_report=analysis_report,
                    html_results=html_results,
                    max_attempts=max_attempts
                ))

                code = result.output.data_analysis_pythonic_code
                explanation = result.output.explanation

                # Add generated code to memory
                memory.append({"role": "assistant", "content": code})

                logfire.info(
                    f"Generated code for attempt {attempt}", code_length=len(code))

                # Execute the generated code
                execution_result = await execute_code(code, temp_dir, attempt)
                execution_results.append(execution_result)

                logfire.info(
                    f"Code execution completed (attempt {attempt})",
                    success=execution_result.success,
                    has_output=bool(execution_result.output),
                    error=execution_result.error[:200] if execution_result.error else None
                )

                if execution_result.success:
                    # Success! Parse and return results
                    logfire.info(
                        f"Answer generation successful on attempt {attempt}")

                    # Try to parse output as JSON if possible
                    final_answer = execution_result.output
                    try:
                        # Attempt to parse as JSON
                        if execution_result.output:
                            json_output = json.loads(
                                execution_result.output.strip())
                            final_answer = json_output
                    except (json.JSONDecodeError, AttributeError):
                        # Keep as string if not valid JSON
                        final_answer = execution_result.output

                    return {
                        "status": "success",
                        "final_answer": final_answer,
                        "execution_results": [r.model_dump() for r in execution_results],
                        "code_generated": code,
                        "attempts_used": attempt,
                        "processing_summary": f"Successfully generated answer on attempt {attempt}",
                        "explanation": explanation
                    }
                else:
                    # Add error to memory for next iteration
                    error_msg = f"Execution failed: {execution_result.error}"
                    memory.append({"role": "user", "content": error_msg})
                    logfire.warn(f"Attempt {attempt} failed",
                                 error=execution_result.error)

            except Exception as e:
                error_msg = f"Code generation failed: {str(e)}"
                logfire.error(f"Attempt {attempt} exception", error=str(e))

                execution_results.append(CodeExecutionResult(
                    success=False,
                    error=error_msg
                ))

                # Add error to memory for next iteration
                memory.append({"role": "user", "content": error_msg})

        # All attempts failed
        logfire.error("All answer generation attempts failed")
        return {
            "status": "error",
            "final_answer": None,
            "execution_results": [r.model_dump() for r in execution_results],
            "code_generated": None,
            "attempts_used": max_attempts,
            "error_message": "All code generation attempts failed",
            "processing_summary": f"Failed after {max_attempts} attempts"
        }
