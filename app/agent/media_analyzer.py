from pydantic import BaseModel, Field
from app.config import get_gemini_model
from pydantic_ai import Agent, BinaryContent, StructuredDict
from typing import List, Optional
import os
import mimetypes
import aiofiles
import asyncio
import json

# ----------------------------
#   Pydantic Output Schema
# ----------------------------


class MediaMetadata(BaseModel):
    """Structured description of extracted metadata & insights from media."""
    filename: str = Field(...,
                          description="Original filename of the media file")
    media_type: str = Field(...,
                            description="Detected media type: image, audio, video")
    description: str = Field(...,
                             description="High-level semantic description")
    detected_objects: List[str] = Field(
        default_factory=list, description="List of distinct entities, objects, or features present")
    textual_content: Optional[str] = Field(
        None, description="Detected text (OCR or embedded captions)")
    statistical_info: Optional[str] = Field(
        None, description="Any quantifiable data (counts, measurements, dimensions)")
    possible_analytical_uses: List[str] = Field(
        default_factory=list, description="Ways this media could be used in a data analysis pipeline")


# Optional: get JSON schema from Pydantic
mediametadata_schema = MediaMetadata.model_json_schema()


# Create Agent function to ensure it's created in the correct event loop
def create_agent():
    """Create agent with proper event loop context"""
    return Agent(
        output_type=StructuredDict(mediametadata_schema),
        model=get_gemini_model()
    )


# ----------------------------
#   Helper: Read File as Bytes
# ----------------------------


async def _read_file_as_bytes(path: str) -> bytes:
    try:
        async with aiofiles.open(path, "rb") as f:
            return await f.read()
    except Exception as e:
        raise IOError(f"Failed to read file {path}: {str(e)}")

# ----------------------------
#   Helper: Construct Analysis Prompt
# ----------------------------


async def _construct_prompt(fname: str, media_type: str) -> str:
    prompt = f"""
You are a highly skilled data analyst specializing in extracting structured, data-ready insights from {media_type} content.
Analyze the file '{fname}' and produce valid JSON that matches the required schema.

Please extract the following information:
- filename: The original filename
- media_type: The type of media (image, audio, or video)
- description: A high-level semantic description of the content
- detected_objects: List of distinct entities, objects, or features present
- textual_content: Any detected text (OCR or embedded captions) if present
- statistical_info: Any quantifiable data like counts, measurements, dimensions
- possible_analytical_uses: Ways this media could be used in data analysis

Return the analysis as a properly structured JSON object.
"""
    return prompt


# ----------------------------
#   Main Analyzer
# ----------------------------


async def analyze_media(file_path: str) -> Optional[MediaMetadata]:
    print(f"*************Analyzing media file: {file_path}")
    mime_type, _ = mimetypes.guess_type(str(file_path))
    print(f"Detected mime_type for {file_path}: {mime_type}")
    if not mime_type:
        mime_type = "application/octet-stream"

    if mime_type.startswith("image"):
        print(f"Processing image file: {file_path}")
        media_type = "image"
    elif mime_type.startswith("audio"):
        print(f"Processing audio file: {file_path}")
        media_type = "audio"
    elif mime_type.startswith("video"):
        print(f"Processing video file: {file_path}")
        media_type = "video"
    else:
        print(f"Skipping unsupported file type for {file_path}: {mime_type}")
        return None

    try:
        # Create agent within the function to ensure proper event loop context
        agent = create_agent()

        file_bytes = await _read_file_as_bytes(str(file_path))
        print(f"Read {len(file_bytes)} bytes from {file_path}")
        media_content = BinaryContent(data=file_bytes, media_type=mime_type)

        prompt = await _construct_prompt(str(file_path), media_type)
        print(f"Generated prompt for {file_path}: {prompt[:200]}...")

        result = await agent.run([
            prompt, media_content
        ])

        print(f"Raw agent output for {file_path}: {result.output}")
        print(f"Output type: {type(result.output)}")

        if not isinstance(result.output, dict):
            print(
                f"ERROR: Agent output is not a dict for {file_path}. Got {type(result.output)}")
            return None

        try:
            metadata = MediaMetadata(**result.output)
            print(f"Successfully serialized MediaMetadata for {file_path}")
            return metadata
        except Exception as serialize_error:
            print(f"SERIALIZATION ERROR for {file_path}: {serialize_error}")
            print(f"Raw output was: {result.output}")
            return None

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None
