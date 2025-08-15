import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import traceback
import logfire

from app.agent.file_manager import create_temp_dir, save_multiple_files, cleanup_temp_dir

from app.agent.orchestrater import orchestrator

# Configure Logfire
logfire.configure()


async def handle_payload_and_response(payload: Dict[str, Any]) -> Any:
    """
    Optimized main payload handler with centralized temp directory management

    Args:
        payload: Dictionary containing data_analyst_input and optional attachments

    Returns:
        calls orchestrater_agent with uploaded files and temp directory
    """

    try:
        data_analyst_input = payload["data_analyst_input"]
        attachments = payload.get("attachments", [])
    except KeyError as e:
        return {"status": "error", "error": f"Missing required field: {e}"}

    with logfire.span("handle_payload_and_response") as span:
        span.set_attribute("input_length", len(data_analyst_input))
        span.set_attribute("attachment_count", len(attachments))

        logfire.info("Processing payload")

        # Centralized temp directory creation - single source of truth
        tmp_dir = await create_temp_dir(data_analyst_input)

        # Step 1: Handle file uploads if any
        uploaded_files_paths: List[Path] = []
        if attachments:
            uploaded_files_paths = await save_multiple_files(attachments, tmp_dir)
            logfire.info("Files saved", temp_dir=str(
                tmp_dir), file_count=len(uploaded_files_paths))
        else:
            logfire.info("No attachments provided")

        orchestrator_response = await orchestrator(
            uploaded_files_paths=uploaded_files_paths,
            data_analysis_input=data_analyst_input,
            temp_dir=str(tmp_dir)
        )
        logfire.info("Orchestrator completed", response_type=type(
            orchestrator_response).__name__)

        # Log the response structure without sensitive data
        if isinstance(orchestrator_response, dict):
            logfire.info("Orchestrator response structure",
                         keys=list(orchestrator_response.keys()),
                         status=orchestrator_response.get("status"))
        else:
            logfire.info("Orchestrator returned non-dict response")

        # if successful, cleanup temp dir
        # await cleanup_temp_dir(tmp_dir)

        logfire.info("Returning orchestrator response to client")
        return orchestrator_response
