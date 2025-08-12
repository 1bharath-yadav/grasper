from pydantic import BaseModel, Field
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider
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

# ----------------------------
#   Gemini Flash LLM Setup
# ----------------------------
gemini_key = os.getenv("GEMINI_API_KEY")
if not gemini_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

settings = GoogleModelSettings(google_thinking_config={"thinking_budget": 0})
provider = GoogleProvider(api_key=gemini_key)
model = GoogleModel("gemini-2.5-flash", provider=provider)

# Create Agent with model
agent = Agent(
    output_type=StructuredDict(mediametadata_schema),
    model=model
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


def _construct_prompt(fname: str, media_type: str) -> str:
    return f"""
You are a highly skilled data analyst specializing in extracting structured, data-ready insights from {media_type} content.
Analyze the file '{fname}' and produce valid JSON."""


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
        file_bytes = await _read_file_as_bytes(str(file_path))
        print(f"Read {len(file_bytes)} bytes from {file_path}")
        media_content = BinaryContent(data=file_bytes, media_type=mime_type)

        result = await agent.run([
            _construct_prompt(str(file_path), media_type), media_content
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
            print(
                f"Required fields: {list(MediaMetadata.model_fields.keys())}")
            print(
                f"Output keys: {list(result.output.keys()) if isinstance(result.output, dict) else 'Not a dict'}")
            return None

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


# # Example usage:
# # Analyze all files in the directory
# async def test_analyze_media():
#     media_dir = "/home/archer/projects/grasper/tests/network"
#     files = os.listdir(media_dir)
#     results = []
#     for file in files:
#         if file.endswith(('.jpg', '.jpeg', '.png', '.gif', '.mp4', '.avi', '.mp3', '.wav')):
#             file_path = os.path.join(media_dir, file)
#             result = await analyze_media(file_path)
#             results.append(result)
#     return results

# # Run the test
# results = asyncio.run(test_analyze_media())
# print(results)
