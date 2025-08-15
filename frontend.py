# app.py
import base64
import json
import time
from io import BytesIO, StringIO
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Grasper - Data Analysis Tool",
    page_icon="https://lobehub.com/icons/gemma",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Helper functions ----------


def safe_decode_base64(data: str) -> bytes:
    """
    Handles strings that may be 'data:image/png;base64,...' or raw base64 strings.
    """
    if data.startswith("data:"):
        # data:<mime>;base64,<base64data>
        try:
            header, b64 = data.split(",", 1)
            return base64.b64decode(b64)
        except Exception:
            pass
    # fallback: try raw base64
    return base64.b64decode(data)


def is_base64_string(s: str) -> bool:
    try:
        # quick test: decode then re-encode and compare (works for common cases)
        _ = base64.b64decode(s, validate=True)
        return True
    except Exception:
        return False


def preview_file(uploaded_file) -> None:
    """
    Show a small preview depending on file type.
    """
    name = uploaded_file.name.lower()
    try:
        uploaded_file.seek(0)
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head(10))
        elif name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
            st.dataframe(df.head(10))
        elif name.endswith(".json"):
            uploaded_file.seek(0)
            obj = json.load(uploaded_file)
            st.json(obj, expanded=False)
        elif name.endswith(".txt"):
            uploaded_file.seek(0)
            txt = uploaded_file.read().decode("utf-8", errors="ignore")
            st.text_area("Text preview", value=txt[:2000], height=200)
        else:
            st.write("No preview available for this file type.")
    except Exception as e:
        st.write("Preview failed:", e)


def make_multipart_files(uploaded_files, questions_text: str, use_questions_file: bool):
    """
    Build files dict for requests.post(files=...)
    """
    files = {}
    # Ensure we provide questions.txt (either uploaded or generated)
    if use_questions_file:
        # find uploaded questions.txt
        for f in uploaded_files:
            if f.name.lower() == "questions.txt":
                f.seek(0)
                files["questions.txt"] = (
                    "questions.txt", f.read(), "text/plain")
                break
    else:
        # create questions.txt from text
        files["questions.txt"] = (
            "questions.txt",
            questions_text.encode("utf-8"),
            "text/plain",
        )

    # Add other uploaded files
    for f in uploaded_files:
        if f.name.lower() == "questions.txt":
            continue
        f.seek(0)
        content = f.read()
        mime = f.type or "application/octet-stream"
        files[f.name] = (f.name, content, mime)
    return files


def display_metrics_and_visuals(parsed: Dict[str, Any]):
    # Large images area first, then metrics/tables
    # Collect images
    images = []
    metrics = {}
    tables = {}

    for k, v in parsed.items():
        lname = k.lower()
        if isinstance(v, str) and len(v) > 50 and is_base64_string(v.split(",")[-1]):
            # candidate base64 image string
            images.append((k, v))
        elif (
            lname.endswith("_chart")
            or lname.endswith("_graph")
            or lname.endswith("_image")
            or (isinstance(v, str) and v.startswith("data:image/"))
        ):
            images.append((k, v))
        elif isinstance(v, (int, float, str)) and not (
            isinstance(v, str) and len(v) > 500
        ):
            metrics[k] = v
        elif isinstance(v, list):
            # maybe tabular data
            try:
                df = pd.DataFrame(v)
                tables[k] = df
            except Exception:
                metrics[k] = v
        elif isinstance(v, dict):
            # small dict => metrics; bigger dict might be nested table
            # Heuristic: if dict has lists of similar length -> try df
            values = list(v.values())
            if all(isinstance(x, (list, tuple)) for x in values):
                try:
                    df = pd.DataFrame(v)
                    tables[k] = df
                except Exception:
                    metrics[k] = v
            else:
                # flatten for metrics
                metrics[k] = v

    if metrics:
        st.subheader("📈 Metrics")
        cols = st.columns(3)
        i = 0
        for k, v in metrics.items():
            with cols[i % 3]:
                st.metric(label=k.replace("_", " ").title(), value=str(v))
            i += 1

    if tables:
        st.subheader("📋 Tables")
        for name, df in tables.items():
            st.markdown(f"**{name.replace('_', ' ').title()}**")
            st.dataframe(df)
            # allow download
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                f"⬇️ Download `{name}.csv`",
                data=csv,
                file_name=f"{name}.csv",
                mime="text/csv",
            )

    if images:
        st.subheader("📊 Visualizations")
        for name, b64 in images:
            try:
                # handle possible data: header
                raw = (
                    b64.split(",")[-1]
                    if "," in b64 and b64.startswith("data:")
                    else b64
                )
                img_bytes = safe_decode_base64(raw)
                img = Image.open(BytesIO(img_bytes))
                st.markdown(f"**{name.replace('_', ' ').title()}**")
                st.image(img, use_container_width=True)
            except Exception as e:
                st.write(f"Could not render image `{name}`: {e}")


# ---------- UI Layout ----------


st.title("📊 Grasper - Data Analysis Tool")
st.markdown(
    "Upload data files and ask questions to get instant analyses, visualizations and downloadable results."
)

DEFAULT_API = "http://localhost:8000/api"

with st.sidebar:
    st.header("⚙️ Configuration")
    api_endpoint = st.text_input(
        "API Endpoint",
        value=DEFAULT_API,
        help="URL of the Grasper API endpoint (POST /api)",
    )
    timeout = st.number_input(
        "Request timeout (seconds)", min_value=30, value=120, step=10
    )
    show_raw = st.checkbox("Show raw API response by default", value=False)
    st.markdown("---")
    st.subheader("💡 Sample Questions")
    st.markdown(
        """
        **For Sales Data:**\n
        - What is the total sales across all regions?\n
        - Which region has the highest sales?\n
        - Plot sales by region as a bar chart\n\n
        **For Network Data:**\n
        - How many edges are in the network?\n
        - Which node has the highest degree?\n        """
    )

# Main columns
left, right = st.columns([1, 1])

with left:
    st.subheader("📝 Analysis Request")

    # Supported extensions from config
    SUPPORTED_EXTS = [
        "zip",
        "tar",
        "tgz",
        "tar.gz",
        "tar.bz2",
        "tar.xz",  # ARCHIVE_EXTS
        "csv",
        "tsv",
        "txt",
        "json",
        "jsonl",
        "parquet",
        "pq",
        "feather",
        "arrow",
        "xls",
        "xlsx",
        "xlsm",
        "html",
        "htmlx",
        "htm",  # DATA_FILE_EXTS
        "png",
        "jpg",
        "jpeg",
        "gif",
        "bmp",
        "tiff",
        "mp3",
        "wav",
        "mp4",
        "avi",
        "mov",
        "mkv",
        "webm",
        "pdf",  # MEDIA_EXTS
    ]
    uploaded_files = st.file_uploader(
        "Upload Data, Media, or Archive Files. Include questions.txt to auto-fill questions.",
        type=SUPPORTED_EXTS,
        accept_multiple_files=True,
    )

    questions_from_file = ""
    questions_file_found = False

    if uploaded_files:
        for f in uploaded_files:
            if f.name.lower() == "questions.txt":
                try:
                    f.seek(0)
                    questions_from_file = f.read().decode("utf-8")
                    questions_file_found = True
                    st.success(
                        "✅ Found questions.txt — loaded into questions box.")
                except Exception:
                    st.warning(
                        "Could not read uploaded questions.txt (encoding issue)."
                    )

    questions = st.text_area(
        "Questions / Instructions",
        value=(questions_from_file if questions_file_found else ""),
        height=220,
        placeholder="Write your analysis questions here, or upload questions.txt",
    )

    st.markdown("**Uploaded files preview**")
    if uploaded_files:
        for f in uploaded_files:
            with st.expander(f.name, expanded=False):
                st.write(f"Type: {f.type} — Size: {f.size} bytes")
                preview_file(f)
    else:
        st.info("No files uploaded yet.")

    use_questions_file = st.checkbox(
        "Use uploaded questions.txt (if present)", value=questions_file_found
    )

    analyze_button = st.button("🚀 Analyze Data", type="primary")

with right:
    st.subheader("📊 Results & Output")
    # History area
    if "history" not in st.session_state:
        st.session_state.history = []

    if analyze_button:
        # Validation
        if not questions.strip() and not questions_file_found:
            st.error(
                "Please enter your analysis questions or upload a `questions.txt` file."
            )
        else:
            # Build multipart files (allow empty uploaded_files)
            files = make_multipart_files(
                uploaded_files if uploaded_files else [], questions, use_questions_file
            )

            status = st.empty()
            progress = st.progress(0)

            try:
                status.info("Sending request to API...")
                progress.progress(10)
                # send request
                with st.spinner("Awaiting API response..."):
                    resp = requests.post(
                        api_endpoint, files=files, timeout=timeout)
                progress.progress(60)
                if resp.status_code != 200:
                    status.error(
                        f"API returned {resp.status_code}: {resp.text}")
                    progress.progress(100)
                else:
                    # parse JSON
                    try:
                        result = resp.json()
                    except Exception:
                        # Could be text or something else
                        st.error("API returned non-JSON response.")
                        st.text(resp.text)
                        result = None

                    progress.progress(80)
                    status.success("API processed the request.")

                    if result is not None:
                        # store to history (limit size)
                        st.session_state.history.insert(
                            0, {"time": time.time(), "result": result}
                        )
                        st.session_state.history = st.session_state.history[:10]

                        if show_raw:
                            with st.expander("🔍 Raw response (JSON)", expanded=False):
                                st.json(result)

                        # attempt structured display using your API's expected shape
                        # Common shape: { "answers": { "answer": <str|dict>, "execution_time": float, "generated_code": str } }
                        answers = None
                        if isinstance(result, dict) and "answers" in result:
                            answers = result["answers"]
                        elif isinstance(result, dict) and "answer" in result:
                            # some APIs use top-level 'answer'
                            answers = {"answer": result["answer"]}
                        else:
                            # fallback: treat entire result as answer
                            answers = {"answer": result}

                        # Show generated code if present
                        if (
                            isinstance(answers, dict)
                            and "generated_code" in answers
                            and answers["generated_code"]
                        ):
                            with st.expander(
                                "🐍 Generated Python Code", expanded=False
                            ):
                                st.code(answers["generated_code"],
                                        language="python")
                                # also offer download
                                generated_code_str = answers["generated_code"]
                                if not isinstance(generated_code_str, str):
                                    generated_code_str = json.dumps(
                                        generated_code_str, indent=2
                                    )
                                st.download_button(
                                    "⬇️ Download generated code",
                                    data=generated_code_str.encode("utf-8"),
                                    file_name="generated_code.py",
                                    mime="text/x-python",
                                )

                        # Primary answer body (string or dict)
                        body = (
                            answers.get("answer")
                            if isinstance(answers, dict)
                            else answers
                        )

                        # If it's a long string that looks like JSON dict, try to parse
                        parsed_answer = None
                        if isinstance(body, str):
                            # try JSON
                            try:
                                parsed_answer = json.loads(body)
                            except Exception:
                                # try eval for dict-like strings (use with caution)
                                try:
                                    if body.strip().startswith(
                                        "{"
                                    ) and body.strip().endswith("}"):
                                        parsed_answer = eval(
                                            body, {"__builtins__": {}})
                                except Exception:
                                    parsed_answer = None
                        elif isinstance(body, (dict, list)):
                            parsed_answer = body

                        if parsed_answer is None:
                            # Just display as text / string
                            st.subheader("💬 Answer")
                            st.write(body)
                        else:
                            # If parsed_answer is a list of dicts -> show as dataframe
                            if isinstance(parsed_answer, list):
                                try:
                                    df = pd.DataFrame(parsed_answer)
                                    st.subheader("📋 Result Table (parsed)")
                                    st.dataframe(df)
                                    csv = df.to_csv(
                                        index=False).encode("utf-8")
                                    st.download_button(
                                        "⬇️ Download CSV",
                                        data=csv,
                                        file_name="result.csv",
                                        mime="text/csv",
                                    )
                                except Exception:
                                    st.write(parsed_answer)
                            elif isinstance(parsed_answer, dict):
                                display_metrics_and_visuals(parsed_answer)
                            else:
                                st.write(parsed_answer)

                    progress.progress(100)

            except requests.exceptions.Timeout:
                status.error(
                    "⏱️ Request timed out. Try increasing the timeout in the sidebar."
                )
                progress.progress(100)
            except requests.exceptions.ConnectionError:
                status.error(
                    "🔌 Could not connect to the API. Is the server running and reachable?"
                )
                progress.progress(100)
            except Exception as e:
                status.error(f"❌ Unexpected error: {e}")
                progress.progress(100)
            finally:
                time.sleep(0.3)
                progress.empty()
                status.empty()

    # Show last 3 responses quick history
    st.markdown("---")
    st.subheader("🕘 Recent Results")
    if st.session_state.history:
        for idx, entry in enumerate(st.session_state.history[:3]):
            t = time.strftime("%Y-%m-%d %H:%M:%S",
                              time.localtime(entry["time"]))
            with st.expander(f"{idx+1}. Result at {t}", expanded=False):
                st.json(entry["result"])
                # offer download of entire JSON
                st.download_button(
                    f"⬇️ Download JSON #{idx+1}",
                    data=json.dumps(entry["result"], indent=2).encode("utf-8"),
                    file_name=f"result_{idx+1}.json",
                    mime="application/json",
                )
    else:
        st.info("No recent results yet. Run an analysis to populate history.")

# Footer / Quick start
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#666'>🚀 Powered by Grasper | Built with Streamlit</div>",
    unsafe_allow_html=True,
)

if not uploaded_files and not questions.strip():
    st.subheader("🚀 Quick Start")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            "**1. Write Questions**\n- State the metrics or plots you want\n- Ask specific questions"
        )
    with col2:
        st.markdown(
            "**2. Upload Files**\n- CSV, Excel, JSON, TXT supported\n- Include `questions.txt` to auto-fill questions"
        )
    with col3:
        st.markdown(
            "**3. Analyze**\n- Press Analyze and wait for results\n- Download CSVs or code as needed"
        )
