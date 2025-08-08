from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from typing import List, Optional
from pathlib import Path
from app.payload_handler import handle_payload
from app.agent.executor import BASE_TMP

router = APIRouter()


@router.post("/analyze", tags=["Analysis"])
async def analyze_data(
    data_analyst_input: UploadFile = File(..., alias="questions.txt"),
    attachments: Optional[List[UploadFile]] = File(default=[])
):
    # Read full text as-is
    input_text = (await data_analyst_input.read()).decode("utf-8")

    # Treat as a single text input
    payload = {
        "data_analyst_input": input_text,
        "attachments": attachments
    }

    response = await handle_payload(payload)
    return response


@router.get("/files", tags=["Files"])
async def list_temp_directories():
    """List all temporary directories with their contents"""
    try:
        if not BASE_TMP.exists():
            return {"directories": [], "message": "No temp directory found"}

        directories = []
        for dir_path in BASE_TMP.iterdir():
            if dir_path.is_dir():
                files = []
                for file_path in dir_path.iterdir():
                    if file_path.is_file():
                        files.append({
                            "name": file_path.name,
                            "size": file_path.stat().st_size,
                            "path": str(file_path.relative_to(BASE_TMP))
                        })
                directories.append({
                    "name": dir_path.name,
                    "files": files,
                    "total_files": len(files)
                })

        return {
            "base_path": str(BASE_TMP),
            "directories": directories,
            "total_directories": len(directories)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error listing files: {str(e)}")


@router.get("/files/{directory}/{filename}", tags=["Files"])
async def download_file(directory: str, filename: str):
    """Download a file from a specific temporary directory"""
    try:
        file_path = BASE_TMP / directory / filename

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        if not file_path.is_file():
            raise HTTPException(status_code=400, detail="Path is not a file")

        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type='application/octet-stream'
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error downloading file: {str(e)}")
