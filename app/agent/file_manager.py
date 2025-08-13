import os
import hashlib
import shutil
from pathlib import Path
from typing import Union, List, Any
import aiofiles
import asyncio

from app.config import TEMP_DIR

# Use config temp directory
BASE_TMP = Path(TEMP_DIR)


def get_hashed_dirname(data_analyst_input: str) -> str:
    """Generate a consistent hash-based directory name from input"""
    return hashlib.sha256(data_analyst_input.encode()).hexdigest()[:10]


async def create_temp_dir(data_analyst_input: str) -> Path:
    """Create a temporary directory based on the input hash"""
    hashed_dirname = get_hashed_dirname(data_analyst_input)
    tmp_dir = BASE_TMP / hashed_dirname

    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir


async def save_uploaded_file(uploaded_file: Any, temp_dir: Path) -> Path:
    """Save an uploaded file to a temporary directory"""
    try:
        # Handle different file object types
        if hasattr(uploaded_file, 'filename'):
            filename = uploaded_file.filename
        elif hasattr(uploaded_file, 'name'):
            filename = Path(uploaded_file.name).name
        elif isinstance(uploaded_file, (str, Path)):
            # If it's already a path, copy it
            source_path = Path(uploaded_file)
            if source_path.exists():
                dest_path = temp_dir / source_path.name
                shutil.copy2(source_path, dest_path)
                return dest_path
            else:
                raise FileNotFoundError(
                    f"Source file not found: {source_path}")
        else:
            filename = f"unnamed_file_{hash(str(uploaded_file))}"

        file_path = temp_dir / filename
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Handle file content based on type
        if hasattr(uploaded_file, 'read'):
            # File-like object
            if asyncio.iscoroutinefunction(uploaded_file.read):
                # Async file object
                async with aiofiles.open(file_path, 'wb') as f:
                    content = await uploaded_file.read()
                    await f.write(content)
            else:
                # Sync file object
                with open(file_path, 'wb') as f:
                    content = uploaded_file.read()
                    f.write(content)
        else:
            # Handle other types (strings, bytes, etc.)
            content = str(uploaded_file).encode() if isinstance(
                uploaded_file, str) else uploaded_file
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(content)

        return file_path

    except Exception as e:
        raise RuntimeError(
            f"Failed to save file {getattr(uploaded_file, 'filename', str(uploaded_file))}: {e}")


async def save_multiple_files(uploaded_files: List[Any], dest_dir: Path) -> List[Path]:
    """Save multiple uploaded files to a directory"""
    if not uploaded_files:
        return []

    # Process files concurrently but with error handling
    tasks = []
    for file in uploaded_files:
        tasks.append(save_uploaded_file(file, dest_dir))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle results and exceptions
    saved_files = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"[ERROR] Failed to save file {i}: {result}")
        else:
            saved_files.append(result)

    return saved_files


async def cleanup_temp_dir(temp_dir: Union[str, Path]) -> bool:
    """Clean up a temporary directory and all its contents"""
    try:
        temp_path = Path(temp_dir)
        if temp_path.exists() and temp_path.is_dir():
            shutil.rmtree(str(temp_path))
            print(
                f"[DEBUG] Successfully cleaned up temp directory: {temp_path}")
            return True
        else:
            print(
                f"[DEBUG] Temp directory does not exist or is not a directory: {temp_path}")
            return False
    except Exception as e:
        print(f"[ERROR] Failed to clean up temp directory {temp_dir}: {e}")
        return False
