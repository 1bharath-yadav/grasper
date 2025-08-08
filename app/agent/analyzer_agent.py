from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import os
import subprocess
import aiofiles
import asyncio
import pandas as pd
import sqlite3
import hashlib
from concurrent.futures import ThreadPoolExecutor
import time

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext, StructuredDict
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider
import logfire

from .utils import convert_numpy_types

load_dotenv()

# Configure Logfire
logfire.configure()

load_dotenv()

# ---- Agent Dependency Types ----


@dataclass
class AnalyzerDeps:
    temp_dir: str

# ---- Data Model Definitions ----


class FileAnalysis(BaseModel):
    filename: str
    size_bytes: int
    file_type: str
    structure_summary: str
    headers: List[str]
    sample_data: List[Dict[str, Any]]
    total_rows: Optional[int] = None
    encoding: Optional[str] = None


class DirectoryAnalysis(BaseModel):
    total_files: int
    files: List[FileAnalysis]
    summary: str

# ---- LLM Model Setup ----


gemini_key = os.getenv("GEMINI_API_KEY")
if not gemini_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

settings = GoogleModelSettings(google_thinking_config={"thinking_budget": 0})
provider = GoogleProvider(api_key=gemini_key)
model = GoogleModel("gemini-2.5-flash", provider=provider)

# Schema for structured output
analysis_schema = {
    "type": "object",
    "properties": {
        "total_files": {"type": "integer"},
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "size_bytes": {"type": "integer"},
                    "file_type": {"type": "string"},
                    "structure_summary": {"type": "string"},
                    "headers": {"type": "array", "items": {"type": "string"}},
                    "sample_data": {"type": "array"},
                    "total_rows": {"type": "integer"},
                    "encoding": {"type": "string"}
                },
                "required": ["filename", "size_bytes", "file_type", "structure_summary", "headers", "sample_data"]
            }
        },
        "summary": {"type": "string"}
    },
    "required": ["total_files", "files", "summary"]
}

# ---- In-Memory Cache for Analysis Results ----

analysis_cache: Dict[str, Dict[str, Any]] = {}
report_cache: Dict[str, str] = {}

# ---- DataFrame Conversion Functions ----


def csv_to_df(file_path: str) -> pd.DataFrame:
    """Convert CSV to DataFrame with encoding detection."""
    encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode CSV file")


def json_to_df(file_path: str) -> pd.DataFrame:
    """Convert JSON to DataFrame."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        return pd.DataFrame(data)
    elif isinstance(data, dict):
        return pd.DataFrame([data])
    else:
        return pd.DataFrame([{"value": data}])


def excel_to_df(file_path: str) -> pd.DataFrame:
    """Convert Excel to DataFrame."""
    return pd.read_excel(file_path)


def sqlite_to_df(file_path: str) -> pd.DataFrame:
    """Convert SQLite to DataFrame (first table)."""
    conn = sqlite3.connect(file_path)

    # Get first table
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    if not tables:
        conn.close()
        raise ValueError("No tables found in database")

    table_name = tables[0][0]
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df


def xml_to_df(file_path: str) -> pd.DataFrame:
    """Convert XML to DataFrame."""
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(file_path)
        root = tree.getroot()

        data = []
        for elem in root:
            record = {}
            for child in elem:
                record[child.tag] = child.text
            data.append(record)

        return pd.DataFrame(data)
    except Exception:
        raise ValueError("Could not parse XML file")


def parquet_to_df(file_path: str) -> pd.DataFrame:
    """Convert Parquet to DataFrame."""
    return pd.read_parquet(file_path)

# ---- Parallel Analysis Functions ----


def analyze_single_file(file_info: Dict[str, Any], temp_dir: str) -> Dict[str, Any]:
    """Analyze a single file and return structured info."""
    file_path = file_info["path"]
    file_hash = hashlib.md5(
        f"{file_path}_{os.path.getmtime(file_path)}".encode()).hexdigest()

    # Check cache first
    if file_hash in analysis_cache:
        return analysis_cache[file_hash]

    start_time = time.time()
    result = {
        "filename": file_info["name"],
        "size_bytes": file_info["size"],
        "file_type": "Unknown",
        "headers": [],
        "sample_data": [],
        "total_rows": 0,
        "encoding": "unknown",
        "data_types": {},
        "analysis_time": 0,
        "memory_usage": 0
    }

    try:
        df = None
        extension = file_info["extension"]

        # Convert to DataFrame based on file type
        if extension in ['.csv', '.tsv']:
            df = csv_to_df(file_path)
            result["file_type"] = "CSV"
            result["encoding"] = "utf-8"
        elif extension == '.json':
            df = json_to_df(file_path)
            result["file_type"] = "JSON"
            result["encoding"] = "utf-8"
        elif extension in ['.xlsx', '.xls']:
            df = excel_to_df(file_path)
            result["file_type"] = "Excel"
            result["encoding"] = "binary"
        elif extension in ['.db', '.sqlite', '.sqlite3']:
            df = sqlite_to_df(file_path)
            result["file_type"] = "SQLite"
            result["encoding"] = "binary"
        elif extension == '.xml':
            df = xml_to_df(file_path)
            result["file_type"] = "XML"
            result["encoding"] = "utf-8"
        elif extension == '.parquet':
            df = parquet_to_df(file_path)
            result["file_type"] = "Parquet"
            result["encoding"] = "binary"
        else:
            raise ValueError(f"Unsupported file type: {extension}")

        if df is not None:
            # Extract DataFrame information
            result.update({
                "headers": df.columns.tolist(),
                "sample_data": df.head(5).to_dict('records'),
                "total_rows": len(df),
                "data_types": df.dtypes.astype(str).to_dict(),
                "memory_usage": df.memory_usage(deep=True).sum(),
                "null_counts": df.isnull().sum().to_dict(),
                "unique_counts": df.nunique().to_dict()
            })

            # Add statistical summary for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                result["numeric_summary"] = df[numeric_cols].describe().to_dict()

    except Exception as e:
        result["error"] = str(e)

    result["analysis_time"] = time.time() - start_time

    # Cache the result
    analysis_cache[file_hash] = result
    return result


async def write_analysis_report(file_result: Dict[str, Any], temp_dir: str) -> None:
    """Write concise analysis report for a single file."""
    report_path = os.path.join(temp_dir, "analysis_report.txt")

    # Generate concise report
    report_lines = [
        f"=== {file_result['filename']} ===",
        f"Type: {file_result['file_type']} | Size: {file_result['size_bytes']:,} bytes | Rows: {file_result.get('total_rows', 0):,}",
        f"Headers ({len(file_result.get('headers', []))}): {', '.join(file_result.get('headers', [])[:10])}{'...' if len(file_result.get('headers', [])) > 10 else ''}",
    ]

    # Add data types summary
    data_types = file_result.get('data_types', {})
    if data_types:
        type_summary = {}
        for col, dtype in data_types.items():
            simplified_type = str(dtype).split(
                '.')[-1]  # Get last part of dtype
            type_summary[simplified_type] = type_summary.get(
                simplified_type, 0) + 1
        report_lines.append(
            f"Data Types: {', '.join([f'{count} {dtype}' for dtype, count in type_summary.items()])}")

    # Add null/missing data info
    null_counts = file_result.get('null_counts', {})
    if null_counts:
        total_nulls = sum(null_counts.values())
        if total_nulls > 0:
            report_lines.append(
                f"Missing Data: {total_nulls:,} null values across columns")

    # Add sample data preview
    sample_data = file_result.get('sample_data', [])
    if sample_data:
        report_lines.append("Sample Data:")
        for i, row in enumerate(sample_data[:3], 1):
            # Show only first 3 fields of each row to keep it concise
            sample_fields = list(row.items())[:3]
            sample_str = ', '.join(
                [f"{k}: {str(v)[:50]}" for k, v in sample_fields])
            report_lines.append(
                f"  Row {i}: {sample_str}{'...' if len(row) > 3 else ''}")

    # Add error if any
    if 'error' in file_result:
        report_lines.append(f"Error: {file_result['error']}")

    report_lines.append("")  # Empty line between files

    # Write to report file
    async with aiofiles.open(report_path, "a", encoding='utf-8') as f:
        await f.write('\n'.join(report_lines) + '\n')

# ---- Main Analysis Functions ----


async def get_file_paths_info(temp_dir: str) -> List[Dict[str, Any]]:
    """Use subprocess to get all file paths and basic info."""
    try:
        # Use find command to get all files (excluding .py files)
        cmd = ["find", temp_dir, "-type", "f", "!", "-name",
               "*.py", "!", "-name", "analysis_report.txt"]
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True)

        file_paths = result.stdout.strip().split('\n') if result.stdout.strip() else []

        files_info = []
        for file_path in file_paths:
            if file_path:  # Skip empty strings
                try:
                    stat_result = os.stat(file_path)
                    files_info.append({
                        "path": file_path,
                        "name": os.path.basename(file_path),
                        "size": stat_result.st_size,
                        "extension": os.path.splitext(file_path)[1].lower()
                    })
                except OSError:
                    continue

        return files_info

    except subprocess.CalledProcessError:
        return []

# ---- Main Analysis Function ----


async def analyze_directory(temp_dir: str) -> Dict[str, Any]:
    """Analyze all files in the temporary directory using parallel processing."""
    if not os.path.exists(temp_dir):
        return {
            "total_files": 0,
            "files": [],
            "summary": "Directory does not exist"
        }

    # Clear previous analysis report
    report_path = os.path.join(temp_dir, "analysis_report.txt")
    if os.path.exists(report_path):
        os.remove(report_path)

    # Step 1: Get all file paths and basic info using subprocess
    files_info = await get_file_paths_info(temp_dir)

    if not files_info:
        return {
            "total_files": 0,
            "files": [],
            "summary": "No data files found in directory"
        }

    # Step 2: Analyze files in parallel using ThreadPoolExecutor
    analyzed_files = []

    with ThreadPoolExecutor(max_workers=min(len(files_info), 4)) as executor:
        # Submit all analysis tasks
        future_to_file = {
            executor.submit(analyze_single_file, file_info, temp_dir): file_info
            for file_info in files_info
        }

        # Collect results and write reports as they complete
        for future in future_to_file:
            try:
                result = future.result()
                analyzed_files.append(result)

                # Write individual report
                await write_analysis_report(result, temp_dir)

            except Exception as e:
                file_info = future_to_file[future]
                error_result = {
                    "filename": file_info["name"],
                    "size_bytes": file_info["size"],
                    "file_type": "Error",
                    "headers": [],
                    "sample_data": [],
                    "total_rows": 0,
                    "encoding": "unknown",
                    "error": str(e)
                }
                analyzed_files.append(error_result)
                await write_analysis_report(error_result, temp_dir)

    # Step 3: Generate overall summary
    total_size = sum(f.get("size_bytes", 0) for f in analyzed_files)
    total_rows = sum(f.get("total_rows", 0) for f in analyzed_files)
    file_types = list(set(f.get("file_type", "Unknown")
                      for f in analyzed_files))

    summary = f"Analyzed {len(analyzed_files)} files ({total_size:,} bytes, {total_rows:,} total rows). "
    summary += f"Types: {', '.join(file_types)}. "

    if any("error" in f for f in analyzed_files):
        error_count = sum(1 for f in analyzed_files if "error" in f)
        summary += f"{error_count} files had errors."

    # Write summary to report
    summary_lines = [
        "\n" + "="*50,
        "ANALYSIS SUMMARY",
        "="*50,
        summary,
        f"Cache hits: {len([f for f in analyzed_files if f.get('analysis_time', 1) < 0.1])}",
        f"Report saved to: {report_path}",
        "="*50
    ]

    async with aiofiles.open(report_path, "a", encoding='utf-8') as f:
        await f.write('\n'.join(summary_lines) + '\n')

    result = {
        "total_files": len(analyzed_files),
        "files": analyzed_files,
        "summary": summary,
        "report_path": report_path,
        "cache_size": len(analysis_cache)
    }

    # Convert any numpy types to native Python types
    return convert_numpy_types(result)

# ---- Public Interface ----


async def analyze_temp_directory(temp_dir: str) -> Dict[str, Any]:
    """Main function to analyze all files in temp directory and return structured analysis."""
    return await analyze_directory(temp_dir)


def clear_analysis_cache():
    """Clear the analysis cache."""
    global analysis_cache, report_cache
    analysis_cache.clear()
    report_cache.clear()
