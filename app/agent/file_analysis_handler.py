"""
Compact, readable, and effective data file_analysis_handler.

- Robust table reads (CSV/TSV/JSON/Parquet/Excel/feather)
- Handles zip, tar archives (extracts and processes members)
- Optional duckdb / pyarrow accelerations (used if installed)
- Simple JSON cache to skip re-analysis

"""

from __future__ import annotations
import csv
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

# Optional accelerated libs
try:
    import duckdb  # type: ignore
except Exception:
    duckdb = None  # type: ignore

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except Exception:
    pa = None  # type: ignore
    pq = None  # type: ignore

# Core data stack
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

# ---------- config ----------
DATA_FILE_EXTS = {
    ".csv",
    ".tsv",
    ".txt",
    ".json",
    ".jsonl",
    ".parquet",
    ".pq",
    ".feather",
    ".arrow",
    ".xls",
    ".xlsx",
    ".xlsm",
}
ARCHIVE_EXTS = {".zip", ".tar", ".tgz", ".tar.gz", ".tar.bz2", ".tar.xz"}
CACHE_FILENAME = "analysis_cache.json"
MAX_SAMPLE_ROWS = 10
AVOID_FILES = {'.py', 'analysis_summary.json', 'analysis_cache.json'}
# ----------------------------


@dataclass
class FileMeta:
    path: str
    mtime: float
    size: int

    def key(self) -> str:
        return f"{self.path}|{int(self.mtime)}|{self.size}"


def _is_archive(path: Path) -> bool:
    lo = str(path).lower()
    return any(lo.endswith(suf) for suf in ARCHIVE_EXTS)


def _is_data_file(path: Path) -> bool:
    lo = path.suffix.lower()
    return lo in DATA_FILE_EXTS


def _safe_open_text(path: Path, encoding: str = "utf-8"):
    # small helper to try encodings
    try:
        return open(path, "r", encoding=encoding, errors="strict")
    except Exception:
        return open(path, "r", encoding="latin-1", errors="replace")


def _sniff_delim(path: Path, default: str = ",") -> str:
    try:
        with _safe_open_text(path) as f:
            sample = f.read(4096)
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample)
        return dialect.delimiter
    except Exception:
        # fallback: if .tsv extension, use tab; naive fallback
        if path.suffix.lower() == ".tsv":
            return "\t"
        return default


def _extract_archive(archive_path: Path, dst_dir: Path) -> List[Path]:
    extracted = []
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(dst_dir)
            extracted = [dst_dir / f for f in z.namelist()]
    else:
        # tar family
        try:
            with tarfile.open(archive_path, "r:*") as t:
                t.extractall(dst_dir)
                extracted = [dst_dir / m.name for m in t.getmembers()
                             if m.name]
        except Exception:
            # if unsupported, just return empty
            return []
    # Normalize to Path and filter existence
    return [Path(p) for p in extracted if (Path(p).exists())]


def _read_table_sample(path: Path, max_rows: int = MAX_SAMPLE_ROWS, use_duckdb: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Read a small sample of the table (if possible) for schema/preview.
    Returns (df_sample, meta).
    meta contains hints like row_count (estimated or exact if available).
    """
    ext = path.suffix.lower()
    meta: Dict[str, Any] = {}

    # Use duckdb for fast row counting if available and CSV-like
    if duckdb and use_duckdb and ext in {".csv", ".tsv", ".txt"}:
        try:
            con = duckdb.connect(":memory:")
            delimiter = _sniff_delim(path)
            # duckdb can infer header; use read_csv_auto for simplicity
            q_tbl = f"read_csv_auto('{str(path)}')"
            # get row_count without loading into pandas
            result = con.execute(f"SELECT COUNT(*) FROM {q_tbl}").fetchone()
            row_count = int(result[0]) if result is not None else None
            meta["row_count"] = row_count
        except Exception:
            meta["row_count"] = None

    # For sampling read small number of rows using pandas
    try:
        if ext in {".csv", ".txt", ".tsv"}:
            delim = _sniff_delim(path)
            df = pd.read_csv(path, sep=delim, nrows=max_rows, engine="python")
        elif ext in {".json", ".jsonl"}:
            # try json lines first
            with _safe_open_text(path) as f:
                first = f.readline().strip()
                if first.startswith("{") and (path.suffix.lower() == ".jsonl" or "\n" in first or path.suffix.lower() == ".json"):
                    # attempt jsonlines: read up to max_rows lines
                    rows = []
                    with _safe_open_text(path) as fh:
                        for _ in range(max_rows):
                            line = fh.readline()
                            if not line:
                                break
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                rows.append(json.loads(line))
                            except Exception:
                                # fallback to reading whole json
                                rows = []
                                break
                    if rows:
                        df = pd.DataFrame(rows)
                    else:
                        # fallback to reading whole file (maybe an array)
                        df = pd.read_json(path, lines=False, nrows=max_rows)
                else:
                    df = pd.read_json(path, lines=False)
                    df = df.head(max_rows)
        elif ext in {".parquet", ".pq"}:
            if pq and pa:
                # use pyarrow to read a small row group/sample
                table = pq.read_table(
                    str(path), columns=None, use_threads=True)
                df = table.to_pandas().head(max_rows)
                meta["row_count"] = table.num_rows
            else:
                df = pd.read_parquet(path).head(max_rows)
        elif ext in {".feather", ".arrow"}:
            df = pd.read_feather(path).head(max_rows)
        elif ext in {".xls", ".xlsx", ".xlsm"}:
            # read first sheet and sample rows
            df = pd.read_excel(path, sheet_name=0).head(max_rows)
        else:
            # fallback: attempt CSV read
            delim = _sniff_delim(path)
            df = pd.read_csv(path, sep=delim, nrows=max_rows, engine="python")
    except Exception as e:
        # fallback to trying with pandas' more tolerant options
        try:
            df = pd.read_csv(path, engine="python")
            df = df.head(max_rows)
        except Exception:
            # unreadable -> empty df
            df = pd.DataFrame()
    return df, meta


def _summarize_df(df: pd.DataFrame, max_sample_rows: int = MAX_SAMPLE_ROWS) -> Dict[str, Any]:
    """Produce concise summary for a dataframe (schema, missing, sample, basic stats)."""
    if df is None or df.empty:
        return {"rows": 0, "cols": 0, "dtypes": {}, "missing": {}, "sample": []}

    out: Dict[str, Any] = {}
    out["rows"] = int(df.shape[0])
    out["cols"] = int(df.shape[1])
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    out["dtypes"] = dtypes

    # missing counts
    missing = {col: int(df[col].isna().sum()) for col in df.columns}
    out["missing"] = missing

    # basic numeric stats (to keep compact, only for numeric columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    stats = {}
    if numeric_cols:
        descr = df[numeric_cols].describe().to_dict()
        # convert numpy types to python types
        for col, vals in descr.items():
            stats[col] = {k: (float(v) if (v is not None and not pd.isna(
                v)) else None) for k, v in vals.items()}
    out["numeric_stats"] = stats

    # top values for object/categorical columns (up to 5)
    cat_cols = df.select_dtypes(
        include=["object", "category"]).columns.tolist()
    topk = {}
    for col in cat_cols:
        topk_vals = df[col].dropna().astype(
            str).value_counts().head(5).to_dict()
        topk[col] = {str(k): int(v) for k, v in topk_vals.items()}
    out["top_values"] = topk

    # sample rows (as list of dict)
    out["sample"] = df.head(max_sample_rows).fillna(
        "").to_dict(orient="records")

    return out


def _atomic_write_text(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    # ensure parent dir
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)  # atomic on POSIX


def _write_cache_atomic(cache_path: Path, cache_obj: Dict[str, Any]):
    tmp = cache_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache_obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, cache_path)


def _load_cache(cache_path: Path) -> Dict[str, Any]:
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _gather_from_inputs(
    inputs: Union[str, Path, Iterable[Union[str, Path]]],
    extract_archives: bool = True,
) -> Tuple[List[Path], Optional[Path], Path]:
    """
    Accepts:
      - single directory path (str/Path),
      - single file path (str/Path),
      - or an iterable of file/dir paths.

    Returns:
      (files_list, tmp_extract_dir_or_None, base_dir_for_reports)
    Behavior:
      - If a directory is passed it walks it (same rules as before).
      - If a file is passed and it's an archive -> extract it and include extracted members.
      - If a file is passed and is a data file -> include it.
      - base_dir is computed as the common parent of all inputs (used for cache/report).
    """
    # normalize inputs -> list[Path]
    if isinstance(inputs, (str, Path)):
        paths = [Path(inputs)]
    else:
        paths = [Path(p) for p in inputs]

    # compute base_dir as common parent of existing inputs, fallback to cwd
    existing = [str(p.resolve()) for p in paths if p.exists()]
    try:
        common_parent = Path(os.path.commonpath(existing)
                             ) if existing else Path.cwd()
        base_dir = common_parent if common_parent.is_dir() else common_parent.parent
    except Exception:
        base_dir = Path.cwd()

    data_files: List[Path] = []
    temp_extract_dir: Optional[Path] = None

    # Define media extensions (should match those in analyze_data)
    MEDIA_EXTS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff',
                  '.mp3', '.wav', '.mp4', '.avi', '.mov', '.mkv', '.webm', '.pdf'}

    def _add_extracted_members(archive_path: Path):
        nonlocal temp_extract_dir
        if temp_extract_dir is None:
            temp_extract_dir = Path(
                tempfile.mkdtemp(prefix="file_analysis_handler_extract_"))
        members = _extract_archive(archive_path, temp_extract_dir)
        for m in members:
            if m.is_dir():
                for mm in m.rglob("*"):
                    if mm.suffix == ".py" or mm.name in AVOID_FILES:
                        continue
                    if _is_data_file(mm) or mm.suffix.lower() in MEDIA_EXTS:
                        data_files.append(mm)
            else:
                if m.suffix == ".py" or m.name in AVOID_FILES:
                    continue
                if _is_data_file(m) or m.suffix.lower() in MEDIA_EXTS:
                    data_files.append(m)

    # walk inputs
    for p in paths:
        if not p.exists():
            print(f"[DEBUG] Input path does not exist, skipping: {p}")
            continue

        if p.is_dir():
            # walk dir
            for root, dirs, files in os.walk(p):
                dirs[:] = [d for d in dirs if d != ".cache"]
                for fname in files:
                    fp = Path(root) / fname
                    if fp.suffix == ".py" or fp.name in AVOID_FILES:
                        continue
                    if _is_data_file(fp) or fp.suffix.lower() in MEDIA_EXTS:
                        data_files.append(fp)
                    elif _is_archive(fp) and extract_archives:
                        _add_extracted_members(fp)
        else:
            # single file
            if p.suffix == ".py" or p.name in AVOID_FILES:
                continue
            if _is_data_file(p) or p.suffix.lower() in MEDIA_EXTS:
                data_files.append(p)
            elif _is_archive(p) and extract_archives:
                _add_extracted_members(p)

    # normalize unique and sort (resolve to absolute paths)
    files_unique = sorted({p.resolve() for p in data_files})
    try:
        import logfire
        logfire.info(f"[DEBUG] Files gathered for analysis: {files_unique}")
    except Exception:
        print(f"[DEBUG] Files gathered for analysis: {files_unique}")
    return list(files_unique), temp_extract_dir, base_dir


async def analyze_data(
    all_file_paths: List[Path],
    temp_dir: Path,
    *,
    cache_filename: str = CACHE_FILENAME,
    use_duckdb: bool = True,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Analyze data files and return comprehensive report.
    """
    import asyncio  # local import used for media file_analysis_handler runs

    # Use temp_dir for cache/report paths
    cache_path = temp_dir / cache_filename

    # Ensure temp_dir exists
    temp_dir.mkdir(parents=True, exist_ok=True)

    cache = _load_cache(cache_path) if not force else {}
    file_cache = cache.get("files", {})

    report: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_dir": str(temp_dir),
        "files_analyzed": [],
        "stats": {"n_files": len(all_file_paths)},
        "errors": [],
        "data_summary": {},
    }

    updated_file_cache: Dict[str, Any] = {}

    MEDIA_EXTS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff',
                  '.mp3', '.wav', '.mp4', '.avi', '.mov', '.mkv', '.webm', '.pdf'}

    # Process each file
    for p in all_file_paths:
        print(f"[DEBUG] Processing file: {p}")
        try:
            # Check if file exists
            if not p.exists():
                print(f"[DEBUG] File does not exist, skipping: {p}")
                continue

            st = p.stat()
            meta = FileMeta(str(p), st.st_mtime, st.st_size)
            key = meta.key()
            ext = p.suffix.lower()

            # Skip system files
            if p.name in AVOID_FILES or ext == '.py':
                continue

            # Check cache first
            cached_entry = file_cache.get(str(p))
            if cached_entry and cached_entry.get("key") == key and not force:
                print(f"[DEBUG] Using cached analysis for: {p}")
                report["files_analyzed"].append(
                    cached_entry.get("summary", {}))
                updated_file_cache[str(p)] = cached_entry
                continue

            # MEDIA files: run media analysis
            if ext in MEDIA_EXTS:
                print(f"[DEBUG] Detected media file: {p}")
                try:
                    # Import media analyzer if available
                    try:
                        from app.agent.media_analyzer import analyze_media
                        print(f"[DEBUG] Running media analysis for: {p}")

                        # Run async function properly
                        media_result = await analyze_media(str(p))

                        print(
                            f"[DEBUG] Media analysis completed: {media_result}")

                        # Convert MediaMetadata object to dict for JSON serialization
                        if media_result and hasattr(media_result, 'model_dump'):
                            media_result = media_result.model_dump()
                        elif media_result and hasattr(media_result, '__dict__'):
                            media_result = media_result.__dict__
                    except ImportError:
                        media_result = {
                            "error": "Media analyzer not available"}
                except Exception as e:
                    media_result = {"error": repr(e)}
                    print(f"[DEBUG] Media analysis failed for {p}: {e}")

                entry = {
                    "path": str(p.relative_to(temp_dir)) if p.is_relative_to(temp_dir) else str(p),
                    "size": int(st.st_size),
                    "type": "media",
                    "ext": ext,
                    "media_analysis": media_result,
                }
                report["files_analyzed"].append(entry)
                updated_file_cache[str(p)] = {"key": key, "summary": entry}
                print(f"[DEBUG] Added media file to report: {entry['path']}")
                continue

            # DATA files: analyze with pandas
            if _is_data_file(p):
                print(f"[DEBUG] Analyzing data file: {p}")
                try:
                    df_sample, file_meta = _read_table_sample(
                        p, max_rows=MAX_SAMPLE_ROWS, use_duckdb=use_duckdb)
                    summary = _summarize_df(
                        df_sample, max_sample_rows=MAX_SAMPLE_ROWS)

                    # Add file metadata
                    summary.update(file_meta)

                    entry = {
                        "path": str(p.relative_to(temp_dir)) if p.is_relative_to(temp_dir) else str(p),
                        "size": int(st.st_size),
                        "type": "data",
                        "ext": ext,
                        "summary": summary,
                    }

                    report["files_analyzed"].append(entry)
                    updated_file_cache[str(p)] = {"key": key, "summary": entry}
                    print(
                        f"[DEBUG] Added data file to report: {entry['path']}")

                except Exception as e:
                    error_msg = f"Failed to analyze {p}: {repr(e)}"
                    print(f"[DEBUG] {error_msg}")
                    report["errors"].append(error_msg)

                    # Add error entry
                    entry = {
                        "path": str(p.relative_to(temp_dir)) if p.is_relative_to(temp_dir) else str(p),
                        "size": int(st.st_size),
                        "type": "error",
                        "ext": ext,
                        "error": error_msg,
                    }
                    report["files_analyzed"].append(entry)

        except Exception as e:
            error_msg = f"Critical error processing {p}: {repr(e)}"
            print(f"[DEBUG] {error_msg}")
            report["errors"].append(error_msg)

    # Generate overall data summary
    data_files_analyzed = [
        f for f in report["files_analyzed"] if f.get("type") == "data"]
    if data_files_analyzed:
        total_rows = sum(f.get("summary", {}).get("rows", 0)
                         for f in data_files_analyzed)
        total_cols = sum(f.get("summary", {}).get("cols", 0)
                         for f in data_files_analyzed)

        report["data_summary"] = {
            "total_data_files": len(data_files_analyzed),
            "total_rows": total_rows,
            "total_columns": total_cols,
            "file_types": Counter(f.get("ext", "unknown") for f in data_files_analyzed),
        }

    # Update cache
    updated_cache = {"files": updated_file_cache}
    try:
        _write_cache_atomic(cache_path, updated_cache)
    except Exception as e:
        print(f"[DEBUG] Failed to write cache: {e}")

    # Write analysis report
    report_path = temp_dir / "analysis_summary.json"
    try:
        _atomic_write_text(report_path, json.dumps(report, indent=2))
        print(f"[DEBUG] Analysis report written to: {report_path}")
    except Exception as e:
        print(f"[DEBUG] Failed to write analysis report: {e}")

    return report


async def analyze_all_files(temp_dir: str, data_analyst_input: str = "") -> Dict[str, Any]:
    """
    Main entry point for file analysis.

    Args:
        temp_dir: Directory containing files to analyze
        data_analyst_input: User's analysis requirements

    Returns:
        Dictionary containing analysis results
    """
    temp_path = Path(temp_dir)
    print(f"[DEBUG] Starting analysis in temp directory: {temp_path}")

    # Gather all files from the temp directory
    all_files, extract_dir, base_dir = _gather_from_inputs(
        temp_path, extract_archives=True)

    try:
        # Run analysis
        result = await analyze_data(
            all_file_paths=all_files,
            temp_dir=temp_path,
            force=False,
        )

        print(f"[DEBUG] Analysis completed. Found {len(all_files)} files.")
        return result

    finally:
        # Cleanup extracted archives if any
        if extract_dir and extract_dir.exists():
            try:
                shutil.rmtree(extract_dir)
                print(
                    f"[DEBUG] Cleaned up extraction directory: {extract_dir}")
            except Exception as e:
                print(f"[DEBUG] Failed to cleanup extraction directory: {e}")


async def analyze_single_file(file_path: Union[str, Path], temp_dir: str, data_analyst_input: str = "") -> Dict[str, Any]:
    """
    Analyze a single file.

    Args:
        file_path: Path to the file to analyze
        temp_dir: Temporary directory for processing
        data_analyst_input: User's analysis requirements

    Returns:
        Dictionary containing analysis results
    """
    file_path = Path(file_path)
    temp_path = Path(temp_dir)

    if not file_path.exists():
        return {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "base_dir": str(temp_path),
            "files_analyzed": [],
            "stats": {"n_files": 0},
            "errors": [f"File not found: {file_path}"],
        }

    return await analyze_data(
        all_file_paths=[file_path],
        temp_dir=temp_path,
        force=False,
    )


async def analyze_files_list(file_paths: List[Union[str, Path]], temp_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze a list of specific files.

    Args:
        file_paths: List of file paths to analyze
        temp_dir: Optional temporary directory for processing. If None, uses parent dir of first file.

    Returns:
        Dictionary containing analysis results
    """
    if not file_paths:
        return {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "base_dir": str(Path.cwd()),
            "files_analyzed": [],
            "stats": {"n_files": 0},
            "errors": ["No files provided for analysis"],
        }

    # Convert all to Path objects
    path_objects = [Path(f) for f in file_paths]

    # Use provided temp_dir or derive from first file's parent
    if temp_dir:
        temp_path = Path(temp_dir)
    else:
        temp_path = path_objects[0].parent

    # Filter out non-existent files
    existing_files = [f for f in path_objects if f.exists()]
    if not existing_files:
        return {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "base_dir": str(temp_path),
            "files_analyzed": [],
            "stats": {"n_files": 0},
            "errors": [f"No valid files found from: {[str(f) for f in path_objects]}"],
        }

    return await analyze_data(
        all_file_paths=existing_files,
        temp_dir=temp_path,
        force=False,
    )
