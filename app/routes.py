from fastapi import APIRouter, HTTPException, Request, Depends, Response, Cookie
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logfire
from app.payload_handler import handle_payload_and_response
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from dotenv import load_dotenv
import os
import jwt
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional
from pydantic import BaseModel

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 24 * 7

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

# JWT Authentication Functions


def create_jwt_token(user_info: dict) -> str:
    """Create a JWT token with user information"""
    payload = {
        "email": user_info.get("email"),
        "name": user_info.get("name"),
        "picture": user_info.get("picture"),
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS),
        "iat": datetime.now(timezone.utc)
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def verify_jwt_token(token: str) -> dict:
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def get_current_user_from_cookie(auth_token: Optional[str] = Cookie(None)) -> dict:
    """Get current user from JWT cookie"""
    if not auth_token:
        raise HTTPException(
            status_code=401, detail="No authentication token found")

    try:
        user_info = verify_jwt_token(auth_token)
        return user_info
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=401, detail=f"Authentication failed: {str(e)}")

router = APIRouter()
security = HTTPBearer()


@router.post("/login", response_model=LoginResponse, tags=["Authentication"])
async def login(response: Response, credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Login with Google token and receive a JWT cookie for future requests
    """
    try:
        token = credentials.credentials
        id_info = id_token.verify_oauth2_token(
            token,
            google_requests.Request(),
            GOOGLE_CLIENT_ID
        )

        if not id_info.get("email_verified"):
            raise HTTPException(
                status_code=401, detail="Google account email not verified")

        jwt_token = create_jwt_token(id_info)

        expires_at = datetime.now(timezone.utc) + \
            timedelta(hours=JWT_EXPIRE_HOURS)
        response.set_cookie(
            key="auth_token",
            value=jwt_token,
            max_age=JWT_EXPIRE_HOURS * 3600,
            httponly=True,
            secure=True,
            samesite="strict"
        )

        logfire.info("User logged in successfully",
                     user_email=id_info.get("email"))

        return LoginResponse(
            message="Login successful",
            user_email=id_info.get("email"),
            expires_at=expires_at.isoformat()
        )

    except Exception as e:
        logfire.error("Login failed", error=str(e))
        raise HTTPException(status_code=401, detail=f"Login failed: {str(e)}")


@router.post("/logout", tags=["Authentication"])
async def logout(response: Response):
    """
    Logout and clear authentication cookie
    """
    response.delete_cookie(
        key="auth_token",
        httponly=True,
        secure=True,
        samesite="strict"
    )
    logfire.info("User logged out")
    return {"message": "Logout successful"}


@router.get("/me", response_model=UserInfo, tags=["Authentication"])
async def get_current_user(user_info: dict = Depends(get_current_user_from_cookie)):
    """Get current user information from cookie"""
    return UserInfo(
        email=user_info.get("email", ""),
        name=user_info.get("name", ""),
        picture=user_info.get("picture")
    )


@router.post("/set_api_key/", response_model=ApiKeyResponse, tags=["Configuration"])
async def set_api_key(request: ApiKeyRequest):
    """Set Gemini API key for a session"""
    try:
        # Store API key for this session
        api_key_store[request.session_id] = request.api_key
        logfire.info("API key set for session", session_id=request.session_id)

        return ApiKeyResponse(
            message="API key set successfully",
            status="active"
        )
    except Exception as e:
        logfire.error("Failed to set API key", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to set API key")


@router.get("/api_key_status/{session_id}", tags=["Configuration"])
async def get_api_key_status(session_id: str):
    """Check if API key is set for a session"""
    has_key = session_id in api_key_store
    return {
        "session_id": session_id,
        "status": "active" if has_key else "not_set",
        "has_key": has_key
    }


# Dependency function to verify the Google token


async def verify_google_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Verifies the Google ID token from the Authorization header.
    """
    try:
        # Extract the token from the credentials
        token = credentials.credentials

        # Verify the token against Google’s servers and decode it
        id_info = id_token.verify_oauth2_token(
            token,
            google_requests.Request(),
            GOOGLE_CLIENT_ID
        )

        if not id_info.get("email_verified"):
            raise HTTPException(
                status_code=401, detail="Google account email not verified")

        return id_info

    except Exception as e:
        raise HTTPException(
            status_code=401, detail=f"Invalid Google token: {str(e)}")


@router.post("/analyze_data", tags=["Analysis"])
async def analyze_data(request: Request, user_info: dict = Depends(get_current_user_from_cookie)):
    """Handles the analysis request. Protected by JWT Cookie Authentication."""
    logfire.info("API request received", user_email=user_info.get("email"))
    form = await request.form()

    questions_file = None
    attachments = []

    for field_name, field_value in form.items():
        if hasattr(field_value, 'filename') and hasattr(field_value, 'read'):
            if field_name == "questions.txt" or field_value.filename == "questions.txt":  # type: ignore
                questions_file = field_value
            else:
                attachments.append(field_value)

    if not questions_file:
        logfire.error("Questions file missing")
        raise HTTPException(
            status_code=400, detail="questions.txt file is required")

    try:
        input_text = (await questions_file.read()).decode("utf-8")
        logfire.info("Questions file read successfully",
                     input_length=len(input_text))
    except Exception as e:
        logfire.error("Failed to read questions file", error=str(e))
        raise HTTPException(
            status_code=400, detail="Invalid questions.txt file encoding")

    payload = {
        "data_analyst_input": input_text,
        "attachments": attachments
    }

    logfire.info("Calling payload handler", attachment_count=len(attachments))
    result = await handle_payload_and_response(payload)

    logfire.info("API request completed", result_type=type(result).__name__)
    return result


# For official project submission
@router.post("/hero_anand_sir", tags=["Analysis"])
async def analyze_unsecure_data(request: Request):
    logfire.info("API request received")
    form = await request.form()

    questions_file = None
    attachments = []

    for field_name, field_value in form.items():
        if hasattr(field_value, 'filename') and hasattr(field_value, 'read'):
            if field_name == "questions.txt" or field_value.filename == "questions.txt":  # type: ignore
                questions_file = field_value
            else:
                attachments.append(field_value)

    if not questions_file:
        logfire.error("Questions file missing")
        raise HTTPException(
            status_code=400, detail="questions.txt file is required")

    try:
        # pyright: ignore[reportAttributeAccessIssue]
        input_text = (await questions_file.read()).decode("utf-8")
        logfire.info("Questions file read successfully",
                     input_length=len(input_text))

    except Exception as e:
        logfire.error("Failed to read questions file", error=str(e))
        raise HTTPException(
            status_code=400, detail="Invalid questions.txt file encoding")

    payload = {
        "data_analyst_input": input_text,
        "attachments": attachments
    }

    logfire.info("Calling payload handler", attachment_count=len(attachments))
    result = await handle_payload_and_response(payload)

    logfire.info("API request completed", result_type=type(result).__name__)
    return result
