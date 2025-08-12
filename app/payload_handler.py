import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import traceback
import logfire

from app.agent.executor import create_temp_dir, save_multiple_files, cleanup_temp_dir
from app.agent.data_agent import handle_data
from app.agent.analyzer import analyze_data
from app.agent.answer_agent import answer_questions

# Configure Logfire
logfire.configure()


async def handle_payload(payload: Dict[str, Any]) -> Any:
    """
    Optimized main payload handler with centralized temp directory management

    Args:
        payload: Dictionary containing data_analyst_input and optional attachments

    Returns:
        Dictionary containing analysis results or error information
    """
    try:
        data_analyst_input = payload["data_analyst_input"]
        attachments = payload.get("attachments", [])
    except KeyError as e:
        return {"status": "error", "error": f"Missing required field: {e}"}

    with logfire.span("handle_payload") as span:
        span.set_attribute("input_length", len(data_analyst_input))
        span.set_attribute("attachment_count", len(attachments))

        logfire.info(
            "Processing payload",
            input_preview=data_analyst_input[:100] + "..." if len(
                data_analyst_input) > 100 else data_analyst_input,
            attachment_count=len(attachments),
        )

        # Centralized temp directory creation - single source of truth
        tmp_dir = await create_temp_dir(data_analyst_input)

        try:
            # Step 1: Handle file uploads if any
            uploaded_files: List[Path] = []
            if attachments:
                uploaded_files = await save_multiple_files(attachments, tmp_dir)
                logfire.info("Files saved", temp_dir=str(
                    tmp_dir), file_count=len(uploaded_files))
            else:
                logfire.info("No attachments provided")

            # Step 2: Data agent processes input and downloads additional data if needed
            with logfire.span("Downloading data"):
                downloaded_files = await handle_data(data_analyst_input, str(tmp_dir))

                # Normalize downloaded_files to a list
                if downloaded_files is None:
                    downloaded_files = []
                elif isinstance(downloaded_files, str):
                    downloaded_files = [Path(downloaded_files)]
                elif isinstance(downloaded_files, (list, tuple)):
                    downloaded_files = [Path(f) if isinstance(
                        f, str) else f for f in downloaded_files]
                else:
                    downloaded_files = [downloaded_files]

                # Merge uploaded and downloaded file lists
                total_files = uploaded_files + downloaded_files

                logfire.info(
                    "Data downloading completed",
                    total_files_count=len(total_files),
                    total_files_preview=[str(f) for f in total_files[:10]],
                )

            # Step 3: Analyze all files in temp directory
            with logfire.span("data_analysis"):
                logfire.info("Starting file analysis", temp_dir=str(tmp_dir))

                analysis_result = analyze_data(total_files, tmp_dir)

                # Verify analysis report was created
                analysis_report_path = tmp_dir / "analysis_report.txt"
                if not analysis_report_path.exists():
                    raise FileNotFoundError(
                        f"Analysis report was not created at {analysis_report_path}")

                logfire.info(
                    "File analysis completed",
                    total_files=len(total_files),
                    analysis_success=analysis_result.get("success", True),
                )

            # Step 4: Generate answers using the answer agent
            with logfire.span("answer_generation"):
                logfire.info("Generating answers to questions")

                answer_result = await answer_questions(data_analyst_input, str(tmp_dir))

                logfire.info("Answer generation completed")
                return json.dumps(answer_result)

        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            logfire.error("Payload processing failed", error=str(
                e), traceback=traceback.format_exc())
            return {"status": "error", "error": error_msg, "traceback": traceback.format_exc()}

        finally:
            # Configurable cleanup - enable in production
            cleanup_enabled = False  # Set to True in production

            if cleanup_enabled and tmp_dir:
                try:
                    cleanup_success = await cleanup_temp_dir(tmp_dir)
                    logfire.info("Temporary directory cleanup", temp_dir=str(
                        tmp_dir), success=cleanup_success)
                except Exception as cleanup_error:
                    logfire.warning("Cleanup failed", error=str(cleanup_error))
            else:
                logfire.info(
                    "Cleanup skipped - temp directory preserved", temp_dir=str(tmp_dir))
