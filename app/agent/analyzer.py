
"""
Compact, readable, and effective data analyzer.

- Robust table reads (CSV/TSV/JSON/Parquet/Excel/feather)
- Handles zip, tar archives (extracts and processes members)
- Optional duckdb / pyarrow accelerations (used if installed)
- Simple JSON cache to skip re-analysis
- Atomic write of analysis_report.txt suitable for passing to an LLM

Usage:
    python analyzer.py /path/to/data_dir
    from analyzer import analyze_dir
    report = analyze_dir("/path/to/data_dir")
"""

from __future__ import annotations
import argparse
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
    ".html",
    ".htmlx",
    ".htm"
}
ARCHIVE_EXTS = {".zip", ".tar", ".tgz", ".tar.gz", ".tar.bz2", ".tar.xz"}
CACHE_FILENAME = "analysis_cache.json"
REPORT_FILENAME = "analysis_report.txt"
MAX_SAMPLE_ROWS = 10
AVOID_FILES = {'.py', 'analysis_summary.json',
               'analysis_report.txt', 'analysis_cache.json'}
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
            df = pd.read_csv(path)
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

    def _add_extracted_members(archive_path: Path):
        nonlocal temp_extract_dir
        if temp_extract_dir is None:
            temp_extract_dir = Path(
                tempfile.mkdtemp(prefix="analyzer_extract_"))
        members = _extract_archive(archive_path, temp_extract_dir)
        for m in members:
            if m.is_dir():
                for mm in m.rglob("*"):
                    if mm.suffix == ".py" or mm.name in AVOID_FILES:
                        continue
                    if _is_data_file(mm):
                        data_files.append(mm)
            else:
                if m.suffix == ".py" or m.name in AVOID_FILES:
                    continue
                if _is_data_file(m):
                    data_files.append(m)

    # walk inputs
    for p in paths:
        if not p.exists():
            # if user passed a glob-like pattern you may want to expand here, but skip for now
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
                    if _is_data_file(fp):
                        data_files.append(fp)
                    elif _is_archive(fp) and extract_archives:
                        _add_extracted_members(fp)
        else:
            # single file
            if p.suffix == ".py" or p.name in AVOID_FILES:
                continue
            if _is_data_file(p):
                data_files.append(p)
            elif _is_archive(p) and extract_archives:
                _add_extracted_members(p)

    # normalize unique and sort (resolve to absolute paths)
    files_unique = sorted({p.resolve() for p in data_files})
    return list(files_unique), temp_extract_dir, base_dir


def analyze_data(
    all_file_paths: List[Path],
    temp_dir: Path,
    *,
    cache_filename: str = CACHE_FILENAME,
    report_filename: str = REPORT_FILENAME,
    use_duckdb: bool = True,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Analyze data files from a list of file paths.

    Args:
        all_file_paths: List of Path objects pointing to files to analyze
        temp_dir: Path object pointing to the directory where files live and where report should be written
        cache_filename: Name of cache file (default: analysis_cache.json)
        report_filename: Name of report file (default: analysis_report.txt)
        use_duckdb: Whether to use duckdb for optimization (default: True)
        force: Whether to force re-analysis ignoring cache (default: False)

    Returns:
        Dict containing analysis report
    """
    import asyncio  # local import used for media analyzer runs

    # Use temp_dir for cache/report paths
    cache_path = temp_dir / cache_filename
    report_path = temp_dir / report_filename

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
    }

    updated_file_cache: Dict[str, Any] = {}

    MEDIA_EXTS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff',
                  '.mp3', '.wav', '.mp4', '.avi', '.mov', '.mkv', '.webm', 'pdf'}
    HTML_EXTS = {'.html', '.htm'}
    AVOID_FILES = {'.py', 'analysis_summary.json',
                   'analysis_report.txt', 'analysis_cache.json'}

    print(f"[DEBUG] Starting file analysis for {len(all_file_paths)} files")
    print(f"[DEBUG] Files to analyze: {[str(f) for f in all_file_paths]}")
    print(f"[DEBUG] Using temp_dir: {temp_dir}")

    for p in all_file_paths:
        print(f"[DEBUG] Processing file: {p}")
        try:
            print(f"[DEBUG] Processing file: {p}")

            # Check if file exists
            if not p.exists():
                print(f"[DEBUG] File does not exist, skipping: {p}")
                continue

            st = p.stat()
            meta = FileMeta(str(p), st.st_mtime, st.st_size)
            key = meta.key()
            ext = p.suffix.lower()
            print(f"[DEBUG] File extension: {ext}, Size: {st.st_size} bytes")

            # Skip system files
            if p.name in AVOID_FILES or ext == '.py':
                print(f"[DEBUG] Skipping system file: {p}")
                continue

            # MEDIA files: run media analyzer, attach results to report entry
            if ext in MEDIA_EXTS:
                print(f"[DEBUG] Detected media file: {p}")
                try:
                    # call media analyzer (assumes analyze_media is async func)
                    from app.agent.media_analyzer import analyze_media
                    print(f"[DEBUG] Running media analysis for: {p}")
                    media_path = temp_dir / p
                    print(f"[DEBUG] Media path: {media_path}")

                    # Run async function in sync context - handle existing event loop
                    try:
                        # Try to get the current event loop
                        loop = asyncio.get_running_loop()
                        # Create a new thread to run the async function
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(
                                asyncio.run, analyze_media(str(media_path)))
                            media_result = future.result(
                                timeout=30)  # 30 second timeout
                    except RuntimeError:
                        # No running event loop, safe to use asyncio.run()
                        media_result = asyncio.run(
                            analyze_media(str(media_path)))

                    # Convert MediaMetadata object to dict for JSON serialization
                    if media_result and hasattr(media_result, 'model_dump'):
                        media_result = media_result.model_dump()
                    elif media_result and hasattr(media_result, '__dict__'):
                        media_result = media_result.__dict__
                except Exception as e:
                    media_result = {"error": repr(e)}
                    print(f"[DEBUG] Media analysis failed for {p}: {e}")

                entry = {
                    "path": str(p.relative_to(temp_dir)) if p.is_relative_to(temp_dir) else str(p),
                    "size": int(st.st_size),
                    "mtime": int(st.st_mtime),
                    "cached": False,
                    "summary": {"type": "media", "ext": ext},
                    "media_analysis": media_result,
                }
                report["files_analyzed"].append(entry)
                updated_file_cache[str(p)] = {
                    "key": key, "summary": entry["summary"]}
                print(f"[DEBUG] Added media file to report: {entry['path']}")
                continue

            # HTML files: run html analyzer and attach results to report entry
            if ext in HTML_EXTS:
                print(f"[DEBUG] Detected HTML file: {p}")
                try:
                    from app.agent.html_analyzer import analyze_html
                    html_result = analyze_html(str(p))
                except Exception as e:
                    html_result = {"error": repr(e)}
                    print(f"[DEBUG] HTML analysis failed for {p}: {e}")

                entry = {
                    "path": str(p.relative_to(temp_dir)) if p.is_relative_to(temp_dir) else str(p),
                    "size": int(st.st_size),
                    "mtime": int(st.st_mtime),
                    "cached": False,
                    "summary": {"type": "html", "ext": ext},
                    "html_analysis": html_result,
                }
                report["files_analyzed"].append(entry)
                updated_file_cache[str(p)] = {
                    "key": key, "summary": entry["summary"]}
                print(f"[DEBUG] Added HTML file to report: {entry['path']}")
                continue

            # Check if this is a data file
            if not _is_data_file(p):
                print(f"[DEBUG] Skipping non-data file: {p}")
                continue

            print(f"[DEBUG] Processing data file: {p}")

            if (not force) and (file_cache.get(str(p), {}).get("key") == key):
                # reuse cached summary
                summary = file_cache[str(p)]["summary"]
                cached = True
                print(f"[DEBUG] Using cached analysis for: {p}")
            else:
                cached = False
                print(f"[DEBUG] Performing fresh analysis for: {p}")

                # read a small sample to infer schema & compute summary
                sample_df, meta_hints = _read_table_sample(
                    p, max_rows=MAX_SAMPLE_ROWS, use_duckdb=use_duckdb)
                print(
                    f"[DEBUG] Sample data shape: {sample_df.shape if sample_df is not None else 'None'}")
                print(f"[DEBUG] Meta hints: {meta_hints}")

                summary = _summarize_df(
                    sample_df, max_sample_rows=MAX_SAMPLE_ROWS)
                print(
                    f"[DEBUG] Summary generated: {list(summary.keys()) if summary else 'None'}")

                # add meta hints (like row_count if duckdb returned)
                if meta_hints.get("row_count") is not None:
                    summary["estimated_total_rows"] = int(
                        meta_hints["row_count"])
                    print(
                        f"[DEBUG] Added estimated total rows: {summary['estimated_total_rows']}")
                else:
                    # attempt cheap row count only for small files. don't load big files.
                    if summary.get("rows", 0) < 100000:
                        try:
                            full = pd.read_csv(p, nrows=0) if p.suffix.lower() in {
                                ".csv", ".tsv", ".txt"} else None
                        except Exception:
                            full = None

                # include a small preview path
                try:
                    preview_path = str(p.relative_to(temp_dir)) if p.is_relative_to(
                        temp_dir) else str(p)
                except Exception:
                    preview_path = str(p)
                summary["preview_path"] = preview_path

            # record in report
            entry = {
                "path": str(p.relative_to(temp_dir)) if p.is_relative_to(temp_dir) else str(p),
                "size": int(st.st_size),
                "mtime": int(st.st_mtime),
                "cached": bool(cached),
                "summary": summary,
            }
            report["files_analyzed"].append(entry)
            print(
                f"[DEBUG] Added file to report: {entry['path']}, cached: {cached}")

            # update file cache record
            updated_file_cache[str(p)] = {"key": key, "summary": summary}

        except Exception as e:
            error_entry = {"path": str(p), "error": repr(e)}
            report["errors"].append(error_entry)
            print(f"[DEBUG] Error processing file {p}: {e}")
            print(f"[DEBUG] Added error to report: {error_entry}")

    # update and write cache atomically
    new_cache_obj = {
        "generated_at": report["generated_at"], "files": updated_file_cache}
    print(f"[DEBUG] Writing cache with {len(updated_file_cache)} file entries")
    try:
        _write_cache_atomic(cache_path, new_cache_obj)
        print(f"[DEBUG] Successfully wrote cache to: {cache_path}")
    except Exception as e:
        print(f"[DEBUG] Failed to write cache atomically: {e}")
        # best-effort: fallback to non-atomic write
        try:
            cache_path.write_text(json.dumps(
                new_cache_obj, indent=2), encoding="utf-8")
            print(f"[DEBUG] Successfully wrote cache with fallback method")
        except Exception as e2:
            print(f"[DEBUG] Failed to write cache with fallback: {e2}")

    # Compose a human-readable report (compact)
    print(
        f"[DEBUG] Composing human-readable report for {len(report['files_analyzed'])} files")
    lines = []
    lines.append(f"Analysis report for: {report['base_dir']}")
    lines.append(f"Generated: {report['generated_at']}")
    lines.append("")
    lines.append(f"Files found: {report['stats']['n_files']}")
    lines.append("")

    for i, fentry in enumerate(report["files_analyzed"]):
        print(
            f"[DEBUG] Adding file {i+1}/{len(report['files_analyzed'])} to report: {fentry['path']}")
        lines.append(
            f"- {fentry['path']} (size={fentry['size']}, mtime={fentry['mtime']}) cached={fentry.get('cached', False)}")
        summ = fentry.get("summary", {})
        lines.append(
            f"  columns={summ.get('cols', '?')}, rows_sample={summ.get('rows', '?')}, dtypes={list(summ.get('dtypes', {}).items())[:4]}")
        if "estimated_total_rows" in summ:
            lines.append(
                f"  estimated_total_rows={summ['estimated_total_rows']}")

        # show top values for a couple columns if present
        topvals = summ.get("top_values", {})
        if topvals:
            cnt = 0
            for col, tv in topvals.items():
                if cnt >= 2:
                    break
                lines.append(
                    f"  top_values[{col}] = {dict(list(tv.items())[:3])}")
                cnt += 1

        # add sample rows (pretty small)
        sample = summ.get("sample", [])
        if sample:
            lines.append(
                f"  sample_rows (first {min(len(sample), MAX_SAMPLE_ROWS)}):")
            for row in sample[:3]:
                lines.append(
                    "    " + ", ".join(f"{k}={str(v)[:80]}" for k, v in row.items()))

        # If there is HTML or media analysis attached, include a compact/truncated display
        if "html_analysis" in fentry:
            try:
                html_summary = json.dumps(
                    fentry["html_analysis"], indent=2, ensure_ascii=False)
                lines.append("  HTML analysis:")
                lines.extend(
                    ["    " + ln for ln in html_summary.splitlines()])
            except Exception:
                lines.append("  HTML analysis: (could not serialize)")

        if "media_analysis" in fentry:
            try:
                media_summary = json.dumps(
                    fentry["media_analysis"], indent=2, ensure_ascii=False)
                lines.append("  MEDIA analysis:")
                lines.extend(
                    ["    " + ln for ln in media_summary.splitlines()])
            except Exception:
                lines.append("  MEDIA analysis: (could not serialize)")

    if report.get("errors"):
        print(f"[DEBUG] Adding {len(report['errors'])} errors to report")
        lines.append("")
        lines.append("Errors encountered:")
        for e in report["errors"]:
            lines.append(f"  - {e}")

    human_text = "\n".join(lines)
    print(f"[DEBUG] Generated human-readable report with {len(lines)} lines")
    print(f"[DEBUG] Report preview (first 500 chars): {human_text[:500]}...")

    # atomic write the analysis report for LLM consumption
    print(f"[DEBUG] Writing analysis report to: {report_path}")
    try:
        _atomic_write_text(report_path, human_text)
        print(f"[DEBUG] Successfully wrote analysis_report.txt using atomic write")

        # Verify the file was written correctly
        if report_path.exists():
            file_size = report_path.stat().st_size
            print(
                f"[DEBUG] Verification: analysis_report.txt exists, size: {file_size} bytes")
            # Read first few lines to verify content
            with open(report_path, 'r', encoding='utf-8') as f:
                first_lines = [f.readline().strip() for _ in range(3)]
            print(
                f"[DEBUG] Verification: First 3 lines of report: {first_lines}")
        else:
            print(f"[DEBUG] ERROR: analysis_report.txt was not created!")
    except Exception as e:
        print(f"[DEBUG] Atomic write failed: {e}")
        # fallback
        try:
            report_path.write_text(human_text, encoding="utf-8")
            print(
                f"[DEBUG] Successfully wrote analysis_report.txt using fallback method")
        except Exception as e2:
            print(f"[DEBUG] Fallback write also failed: {e2}")

    # also attach the human report string into returned report
    report["human_readable"] = human_text
    print(
        f"[DEBUG] analyze_data completed successfully, returning report with {len(report['files_analyzed'])} files")

    return report

# if __name__ == "__main__":
#     args = _main_args()
#     out = analyze_dir(args.dir, use_duckdb=(
#         not args.no_duckdb), force=args.force)
#     # print combined report for LLM input
#     print(get_combined_report(args.dir))
