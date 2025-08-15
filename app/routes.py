from fastapi import APIRouter, HTTPException, Request, Response
import logfire
from app.payload_handler import handle_payload_and_response
import os
from typing import Optional
from pydantic import BaseModel
from datetime import datetime

# Simplified public-only routes (authentication removed)

router = APIRouter()

# In-memory storage for API keys (session-based)
api_key_store = {}

# Pydantic models


class LoginResponse(BaseModel):
    message: str
    user_email: str
    expires_at: str


class UserInfo(BaseModel):
    email: str
    name: str
    picture: Optional[str] = None


class ApiKeyRequest(BaseModel):
    session_id: str
    api_key: str


class ApiKeyResponse(BaseModel):
    message: str
    status: str


@router.post("/set_api_key/", response_model=ApiKeyResponse, tags=["Configuration"])
async def set_api_key(request: ApiKeyRequest):
    """Set Gemini API key for a session (public).
    """
    try:
        api_key_store[request.session_id] = request.api_key
        logfire.info("API key set for session", session_id=request.session_id)
        return ApiKeyResponse(message="API key set successfully", status="active")
    except Exception as e:
        logfire.error("Failed to set API key", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to set API key")


@router.get("/api_key_status/{session_id}", tags=["Configuration"])
async def get_api_key_status(session_id: str):
    has_key = session_id in api_key_store
    return {"session_id": session_id, "status": "active" if has_key else "not_set", "has_key": has_key}


@router.post("/analyze_data", tags=["Analysis"])
async def analyze_data(request: Request):
    logfire.info("API request received (analyze_data)")
    form = await request.form()

    questions_file = None
    attachments = []

    for field_name, field_value in form.items():
        if hasattr(field_value, 'filename') and hasattr(field_value, 'read'):
            if field_name == "questions.txt" or getattr(field_value, 'filename', '') == "questions.txt":
                questions_file = field_value
            else:
                attachments.append(field_value)

    if not questions_file:
        logfire.error("Questions file missing")
        raise HTTPException(
            status_code=400, detail="questions.txt file is required")

    try:
        content = await questions_file.read()
        if isinstance(content, bytes):
            input_text = content.decode("utf-8")
        else:
            input_text = str(content)
        logfire.info("Questions file read successfully",
                     input_length=len(input_text))
    except Exception as e:
        logfire.error("Failed to read questions file", error=str(e))
        raise HTTPException(
            status_code=400, detail="Invalid questions.txt file encoding")

    payload = {"data_analyst_input": input_text, "attachments": attachments}

    logfire.info("Calling payload handler", attachment_count=len(attachments))
    result = await handle_payload_and_response(payload)

    logfire.info("API request completed", result_type=type(result).__name__)
    return result
