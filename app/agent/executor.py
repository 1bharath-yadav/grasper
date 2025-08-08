import os
import hashlib
import uuid
import shutil
from pathlib import Path
from typing import Union
import aiofiles
import asyncio

# Use absolute path to ensure correct location
BASE_TMP = Path(__file__).parent.parent.parent / "tmp"


def get_hashed_dirname(data_analyst_input: str) -> str:
    # Use SHA256 hash of the input to ensure uniqueness
    return hashlib.sha256(data_analyst_input.encode()).hexdigest()[:10]


async def create_temp_dir(data_analyst_input: str) -> Path:
    # Create a temporary directory based on the input if it doesn't exist if it exists leave it as is
    hashed_dirname = get_hashed_dirname(data_analyst_input)
    tmp_dir = BASE_TMP / hashed_dirname

    if not tmp_dir.exists():
        tmp_dir.mkdir(parents=True, exist_ok=True)

    return tmp_dir


async def move_uploaded_file(src_path: Union[str, Path], dest_dir: Path):
    """Asynchronously move uploaded file into temp dir"""
    dest_path = dest_dir / Path(src_path).name
    shutil.move(str(src_path), str(dest_path))
    return dest_path


async def move_multiple_files(file_paths: list[str], dest_dir: Path) -> list[Path]:
    return await asyncio.gather(
        *[move_uploaded_file(path, dest_dir) for path in file_paths]
    )


async def save_uploaded_file(uploaded_file, temp_dir: Path) -> Path:
    """Save an uploaded file to a temporary directory"""
    try:
        file_path = temp_dir / uploaded_file.filename

        # Ensure the directory exists
        temp_dir.mkdir(parents=True, exist_ok=True)

        print(f"[DEBUG] Saving file: {uploaded_file.filename} to {file_path}")

        # Save the file
        async with aiofiles.open(file_path, 'wb') as f:
            content = await uploaded_file.read()
            await f.write(content)

        print(f"[DEBUG] Successfully saved file: {file_path}")
        print(f"[DEBUG] File size: {file_path.stat().st_size} bytes")

        return file_path
    except Exception as e:
        print(f"[ERROR] Failed to save file {uploaded_file.filename}: {e}")
        raise


async def save_multiple_uploaded_files(uploaded_files: list, dest_dir: Path) -> list[Path]:
    """Save multiple uploaded files to a directory"""
    if not uploaded_files:
        return []

    return await asyncio.gather(
        *[save_uploaded_file(file, dest_dir) for file in uploaded_files]
    )
