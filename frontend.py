import base64
import json
import mimetypes
import time
import uuid
from io import BytesIO, StringIO
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
from PIL import Image

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Configure page
st.set_page_config(
    page_title="Grasper - Advanced Data Analytics",
    page_icon="💤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown(
    """
<style>
    .main > div {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    .stApp > header {
        background-color: transparent;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .main-header {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(45deg, #fff, #e0e6ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 0;
    }
    
    .card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #fff;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        color: white;
        border-bottom: 2px solid rgba(255, 255, 255, 0.2);
        padding-bottom: 0.5rem;
    }
    
    .upload-area {
        border: 2px dashed rgba(255, 255, 255, 0.3);
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: rgba(255, 255, 255, 0.05);
        margin: 1rem 0;
    }
    
    .status-success {
        background: rgba(46, 213, 115, 0.2);
        color: #2ed573;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        border-left: 4px solid #2ed573;
        margin: 1rem 0;
    }
    
    .status-error {
        background: rgba(255, 107, 107, 0.2);
        color: #ff6b6b;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff6b6b;
        margin: 1rem 0;
    }
    
    .status-info {
        background: rgba(116, 185, 255, 0.2);
        color: #74b9ff;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        border-left: 4px solid #74b9ff;
        margin: 1rem 0;
    }
    
    .quick-start {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 2rem;
        margin-top: 2rem;
    }
    
    .step-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .media-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .file-preview {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border-radius: 25px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: 600;
        box-shadow: 0 4px 15px 0 rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(102, 126, 234, 0.6);
    }
    
    .sidebar .stSelectbox > div > div > select {
        background: rgba(255, 255, 255, 0.1);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .sidebar .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.1);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 1px solid rgba(255, 255, 255, 0.2);
        color: rgba(255, 255, 255, 0.7);
    }
</style>
""",
    unsafe_allow_html=True,
)

# API Configuration
# Change this to your FastAPI server URL
API_BASE_URL = "http://localhost:8000"
# API_BASE_URL = "https://bharath4444-grasper-ai.hf.space"  # Uncomment for production

# ========== Helper Functions ==========


def safe_decode_base64(data: str) -> bytes:
    """Handles data URIs and raw base64 strings."""
    if data.startswith("data:"):
        try:
            header, b64 = data.split(",", 1)
            return base64.b64decode(b64)
        except Exception:
            pass
    return base64.b64decode(data)


def is_base64_string(s: str) -> bool:
    """Check if string is valid base64."""
    try:
        base64.b64decode(s, validate=True)
        return True
    except Exception:
        return False


def get_mime_type(data_uri: str) -> str:
    """Extract MIME type from data URI."""
    if data_uri.startswith("data:"):
        try:
            mime_part = data_uri.split(";")[0].replace("data:", "")
            return mime_part
        except Exception:
            return "application/octet-stream"
    return "application/octet-stream"


def render_media_content(name: str, data: str) -> None:
    """Render different types of media content."""
    try:
        mime_type = get_mime_type(data) if data.startswith(
            "data:") else "image/png"

        # Clean data for processing
        if data.startswith("data:"):
            clean_data = data.split(",", 1)[1]
        else:
            clean_data = data

        if not is_base64_string(clean_data):
            st.error(f"Invalid base64 data for {name}")
            return

        decoded_data = base64.b64decode(clean_data)

        st.markdown(
            f'<div class="section-header">{name.replace("_", " ").title()}</div>',
            unsafe_allow_html=True,
        )

        # Handle different media types
        if mime_type.startswith("image/"):
            try:
                img = Image.open(BytesIO(decoded_data))
                st.image(img, use_container_width=True, caption=name)

                # Download button for images
                st.download_button(
                    label="Download Image",
                    data=decoded_data,
                    file_name=f"{name}.{mime_type.split('/')[-1]}",
                    mime=mime_type,
                    key=f"download_{name}_img",
                )
            except Exception as e:
                st.error(f"Could not display image: {e}")

        elif mime_type.startswith("audio/"):
            st.audio(decoded_data, format=mime_type)
            st.download_button(
                label="Download Audio",
                data=decoded_data,
                file_name=f"{name}.{mime_type.split('/')[-1]}",
                mime=mime_type,
                key=f"download_{name}_audio",
            )

        elif mime_type.startswith("video/"):
            st.video(decoded_data, format=mime_type)
            st.download_button(
                label="Download Video",
                data=decoded_data,
                file_name=f"{name}.{mime_type.split('/')[-1]}",
                mime=mime_type,
                key=f"download_{name}_video",
            )

        else:
            # Try to render as image if MIME type detection failed
            try:
                img = Image.open(BytesIO(decoded_data))
                st.image(img, use_container_width=True, caption=name)
                st.download_button(
                    label="Download File",
                    data=decoded_data,
                    file_name=f"{name}.png",
                    mime="image/png",
                    key=f"download_{name}_file",
                )
            except Exception:
                st.error(f"Unsupported media type: {mime_type}")

    except Exception as e:
        st.error(f"Error rendering media {name}: {e}")


def preview_file(uploaded_file) -> None:
    """Show file preview with modern styling."""
    name = uploaded_file.name.lower()

    st.markdown('<div class="file-preview">', unsafe_allow_html=True)

    try:
        uploaded_file.seek(0)
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head(10), use_container_width=True)

        elif name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
            st.dataframe(df.head(10), use_container_width=True)

        elif name.endswith(".json"):
            uploaded_file.seek(0)
            obj = json.load(uploaded_file)
            st.json(obj, expanded=False)

        elif name.endswith(".txt"):
            uploaded_file.seek(0)
            txt = uploaded_file.read().decode("utf-8", errors="ignore")
            st.text_area(
                "Preview", value=txt[:2000], height=200, disabled=True)

        elif name.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
            uploaded_file.seek(0)
            img = Image.open(uploaded_file)
            st.image(img, caption=uploaded_file.name, use_container_width=True)

        else:
            st.info("No preview available for this file type.")

    except Exception as e:
        st.error(f"Preview failed: {e}")

    st.markdown("</div>", unsafe_allow_html=True)


def make_multipart_files(uploaded_files, questions_text: str, use_questions_file: bool):
    """Build files dict for requests."""
    files = {}

    if use_questions_file:
        for f in uploaded_files:
            if f.name.lower() == "questions.txt":
                f.seek(0)
                files["questions.txt"] = (
                    "questions.txt", f.read(), "text/plain")
                break
    else:
        files["questions.txt"] = (
            "questions.txt",
            questions_text.encode("utf-8"),
            "text/plain",
        )

    for f in uploaded_files:
        if f.name.lower() == "questions.txt":
            continue
        f.seek(0)
        content = f.read()
        mime = f.type or "application/octet-stream"
        files[f.name] = (f.name, content, mime)
    return files


def display_results_dashboard(parsed: Dict[str, Any]):
    """Display results in a modern dashboard layout."""

    # Separate content types
    images = []
    media_files = []
    metrics = {}
    tables = {}
    text_content = {}

    for k, v in parsed.items():
        lname = k.lower()

        # Check for base64 encoded content
        if isinstance(v, str) and len(v) > 50:
            if v.startswith("data:"):
                mime_type = get_mime_type(v)
                if mime_type.startswith(("image/", "audio/", "video/")):
                    media_files.append((k, v))
                continue
            elif is_base64_string(v.split(",")[-1] if "," in v else v):
                images.append((k, v))
                continue

        # Check for chart/graph keywords
        if any(
            keyword in lname
            for keyword in ["_chart", "_graph", "_plot", "_image", "_visualization"]
        ):
            if isinstance(v, str):
                images.append((k, v))
                continue

        # Categorize other content
        if isinstance(v, (int, float)):
            metrics[k] = v
        elif isinstance(v, str) and len(v) < 500:
            metrics[k] = v
        elif isinstance(v, list):
            try:
                df = pd.DataFrame(v)
                tables[k] = df
            except Exception:
                text_content[k] = v
        elif isinstance(v, dict):
            # Try to convert to DataFrame
            try:
                df = pd.DataFrame(v)
                tables[k] = df
            except Exception:
                if len(str(v)) < 1000:
                    metrics[k] = str(v)
                else:
                    text_content[k] = v
        else:
            text_content[k] = v

    # Display metrics in cards
    if metrics:
        st.markdown(
            '<div class="section-header">Key Metrics</div>', unsafe_allow_html=True
        )

        # Create responsive columns
        num_metrics = len(metrics)
        cols_per_row = min(4, num_metrics)

        metric_items = list(metrics.items())
        for i in range(0, len(metric_items), cols_per_row):
            cols = st.columns(cols_per_row)

            for j, (key, value) in enumerate(metric_items[i: i + cols_per_row]):
                with cols[j]:
                    st.markdown(
                        f"""
                    <div class="metric-card">
                        <div class="metric-value">{value}</div>
                        <div class="metric-label">{key.replace('_', ' ').title()}</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

    # Display tables
    if tables:
        st.markdown(
            '<div class="section-header">Data Tables</div>', unsafe_allow_html=True
        )

        for name, df in tables.items():
            with st.expander(f"{name.replace('_', ' ').title()}", expanded=True):
                st.dataframe(df, use_container_width=True)

                # Export options
                col1, col2, col3 = st.columns(3)

                with col1:
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download CSV",
                        data=csv,
                        file_name=f"{name}.csv",
                        mime="text/csv",
                        key=f"csv_{name}",
                    )

                with col2:
                    excel_buffer = BytesIO()
                    df.to_excel(excel_buffer, index=False)
                    st.download_button(
                        "Download Excel",
                        data=excel_buffer.getvalue(),
                        file_name=f"{name}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"excel_{name}",
                    )

                with col3:
                    json_data = df.to_json(
                        orient="records", indent=2).encode("utf-8")
                    st.download_button(
                        "Download JSON",
                        data=json_data,
                        file_name=f"{name}.json",
                        mime="application/json",
                        key=f"json_{name}",
                    )

    # Display media content
    if media_files or images:
        st.markdown(
            '<div class="section-header">Visual Content</div>', unsafe_allow_html=True
        )

        # Display media files (audio, video, images with data URIs)
        for name, data in media_files:
            st.markdown('<div class="media-container">',
                        unsafe_allow_html=True)
            render_media_content(name, data)
            st.markdown("</div>", unsafe_allow_html=True)

        # Display base64 images
        for name, data in images:
            st.markdown('<div class="media-container">',
                        unsafe_allow_html=True)
            render_media_content(name, data)
            st.markdown("</div>", unsafe_allow_html=True)

    # Display text content
    if text_content:
        st.markdown(
            '<div class="section-header">Additional Content</div>',
            unsafe_allow_html=True,
        )

        for name, content in text_content.items():
            with st.expander(f"{name.replace('_', ' ').title()}", expanded=False):
                if isinstance(content, (dict, list)):
                    st.json(content)
                else:
                    st.text(str(content))


# ========== Main UI ==========

# Sidebar Configuration
with st.sidebar:
    st.markdown(
        '<div class="section-header">Grasper Analytics</div>', unsafe_allow_html=True
    )

    timeout = st.slider(
        "Request Timeout (seconds)", min_value=30, max_value=300, value=120, step=10
    )

    show_raw_response = st.checkbox("Show raw API response", value=False)
    enable_debug = st.checkbox("Enable debug mode", value=False)

    st.markdown("---")

    # API Key Configuration Section
    st.markdown(
        '<div class="section-header">API Key Setup</div>', unsafe_allow_html=True
    )

    # Initialize session state for session ID if not exists
    if "session_id" not in st.session_state:
        try:
            st.session_state.session_id = str(uuid.uuid4())
        except Exception as e:
            st.error(f"Error initializing session: {e}")
            st.session_state.session_id = f"session_{int(time.time())}"

    # Initialize API key status in session state
    if "api_key_status" not in st.session_state:
        st.session_state.api_key_status = "Not Set"

    if "api_key_value" not in st.session_state:
        st.session_state.api_key_value = ""

    # Display current API key status
    status_color = "🔴" if st.session_state.api_key_status == "Not Set" else "🟢"
    st.markdown(
        f"**Status:** {status_color} {st.session_state.api_key_status}")

    # Input API key from user
    api_key = st.text_input(
        "Enter your Gemini API Key",
        type="password",
        value=st.session_state.api_key_value,
        help="Your Gemini API key will be securely stored for this session only"
    )

    # Button to send key to backend
    col_activate, col_clear = st.columns(2)

    with col_activate:
        if st.button("🔑 Activate API Key", use_container_width=True):
            if api_key:
                try:
                    # Send both API key and session ID to backend
                    response = requests.post(
                        f"{API_BASE_URL}/set_api_key/",
                        json={
                            "session_id": st.session_state.session_id,
                            "api_key": api_key,
                        },
                        timeout=10
                    )
                    if response.status_code == 200:
                        st.session_state.api_key_status = "Active"
                        st.session_state.api_key_value = api_key
                        st.success("✅ API key activated successfully!")
                        st.rerun()
                    else:
                        st.session_state.api_key_status = "Error"
                        st.error(f"❌ Backend error: {response.text}")
                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timed out. Please try again.")
                except requests.exceptions.ConnectionError:
                    st.error(
                        "🌐 Connection error. Check your internet connection.")
                except Exception as e:
                    st.session_state.api_key_status = "Error"
                    st.error(f"❌ Error connecting to backend: {e}")
            else:
                st.warning("⚠️ Please enter an API key.")

    with col_clear:
        if st.button("🗑️ Clear Key", use_container_width=True):
            st.session_state.api_key_status = "Not Set"
            st.session_state.api_key_value = ""
            st.success("🗑️ API key cleared from session.")
            st.rerun()


# Main Content Area
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown(
        '<div class="section-header">Analysis Setup</div>', unsafe_allow_html=True
    )

    # File Upload Section

    uploaded_files = st.file_uploader(
        "Upload any file type",
        type=None,
        accept_multiple_files=True,
        help="You can upload any file type.",
    )

    # Check for questions.txt
    questions_from_file = ""
    questions_file_found = False

    if uploaded_files:
        for f in uploaded_files:
            if f.name.lower() == "questions.txt":
                try:
                    f.seek(0)
                    questions_from_file = f.read().decode("utf-8")
                    questions_file_found = True
                    st.success("Questions loaded from uploaded file!")
                    break
                except Exception:
                    st.warning("Could not read questions.txt file.")

    # Questions Input
    default_question = ""
    if "sample_question" in st.session_state:
        default_question = st.session_state.sample_question
    elif questions_file_found:
        default_question = questions_from_file

    questions = st.text_area(
        "Analysis Questions",
        value=default_question,
        height=200,
        placeholder="Enter your analysis questions here...\n\nExample:\n- What are the key insights from this data?\n- Create visualizations for the main trends\n- Provide summary statistics",
        help="Describe what you want to analyze or discover from your data",
    )

    # File Preview
    if uploaded_files:
        st.markdown(
            '<div class="section-header">File Preview</div>', unsafe_allow_html=True
        )

        for uploaded_file in uploaded_files:
            with st.expander(
                f"{uploaded_file.name} ({uploaded_file.size:,} bytes)", expanded=False
            ):
                preview_file(uploaded_file)

    # Analysis Options
    use_questions_file = st.checkbox(
        "Use uploaded questions.txt file",
        value=questions_file_found,
        disabled=not questions_file_found,
    )

    # Analysis Button
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_button = st.button(
        "Start Analysis",
        type="primary",
        use_container_width=True,
        disabled=not questions.strip(),
    )

with col2:
    st.markdown(
        '<div class="section-header">Results & Insights</div>', unsafe_allow_html=True
    )

    # Initialize session state
    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []

    if analyze_button:
        if not questions.strip():
            st.error("Please enter your analysis questions.")
        else:
            # Prepare request
            files = make_multipart_files(
                uploaded_files or [], questions, use_questions_file
            )

            # Progress tracking
            progress_container = st.container()
            status_container = st.container()

            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()

            try:
                # Send request
                status_text.markdown(
                    '<div class="status-info">Sending request to API...</div>',
                    unsafe_allow_html=True,
                )
                progress_bar.progress(20)

                response = requests.post(
                    f"{API_BASE_URL}/analyze_data", files=files, timeout=timeout)
                progress_bar.progress(60)

                if response.status_code != 200:
                    status_text.markdown(
                        f'<div class="status-error">API Error {response.status_code}: {response.text}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    status_text.markdown(
                        '<div class="status-success">Processing response...</div>',
                        unsafe_allow_html=True,
                    )
                    progress_bar.progress(80)

                    try:
                        result = response.json()

                        # Store in history
                        st.session_state.analysis_history.insert(
                            0,
                            {
                                "timestamp": time.time(),
                                "questions": questions,
                                "result": result,
                                "files": (
                                    [f.name for f in uploaded_files]
                                    if uploaded_files
                                    else []
                                ),
                            },
                        )

                        # Keep only last 10 results
                        st.session_state.analysis_history = (
                            st.session_state.analysis_history[:10]
                        )

                        progress_bar.progress(100)
                        status_text.markdown(
                            '<div class="status-success">Analysis completed successfully!</div>',
                            unsafe_allow_html=True,
                        )

                        # Display results
                        if show_raw_response:
                            with st.expander("Raw API Response", expanded=False):
                                st.json(result)

                        # Process and display structured results
                        answers = None
                        if isinstance(result, dict):
                            if "answers" in result:
                                answers = result["answers"]
                            elif "answer" in result:
                                answers = {"answer": result["answer"]}
                            else:
                                answers = {"answer": result}

                        if answers:
                            # Show generated code if available
                            if (
                                isinstance(answers, dict)
                                and "generated_code" in answers
                            ):
                                code = answers["generated_code"]
                                if code:
                                    with st.expander(
                                        "Generated Python Code", expanded=False
                                    ):
                                        st.code(code, language="python")
                                        st.download_button(
                                            "Download Code",
                                            data=code.encode("utf-8"),
                                            file_name="analysis_code.py",
                                            mime="text/x-python",
                                        )

                            # Display main results
                            answer_content = (
                                answers.get("answer")
                                if isinstance(answers, dict)
                                else answers
                            )

                            # Try to parse structured data
                            parsed_data = None
                            if isinstance(answer_content, str):
                                try:
                                    parsed_data = json.loads(answer_content)
                                except:
                                    try:
                                        if answer_content.strip().startswith("{"):
                                            parsed_data = eval(
                                                answer_content, {
                                                    "__builtins__": {}}
                                            )
                                    except:
                                        pass
                            elif isinstance(answer_content, (dict, list)):
                                parsed_data = answer_content

                            if parsed_data:
                                if isinstance(parsed_data, list):
                                    try:
                                        df = pd.DataFrame(parsed_data)
                                        st.dataframe(
                                            df, use_container_width=True)

                                        csv = df.to_csv(
                                            index=False).encode("utf-8")
                                        st.download_button(
                                            "Download Results as CSV",
                                            data=csv,
                                            file_name="analysis_results.csv",
                                            mime="text/csv",
                                        )
                                    except:
                                        st.json(parsed_data)
                                else:
                                    display_results_dashboard(parsed_data)
                            else:
                                st.markdown("### Analysis Results")
                                st.write(answer_content)

                        # Clear progress indicators after short delay
                        time.sleep(1)
                        progress_container.empty()
                        status_container.empty()

                    except json.JSONDecodeError:
                        status_text.markdown(
                            '<div class="status-error">Invalid JSON response from API</div>',
                            unsafe_allow_html=True,
                        )
                        st.text(response.text)

            except requests.exceptions.Timeout:
                status_text.markdown(
                    '<div class="status-error">Request timeout. Try increasing timeout in settings.</div>',
                    unsafe_allow_html=True,
                )
            except requests.exceptions.ConnectionError:
                status_text.markdown(
                    '<div class="status-error">Connection failed. Check if API server is running.</div>',
                    unsafe_allow_html=True,
                )
            except Exception as e:
                status_text.markdown(
                    f'<div class="status-error">Unexpected error: {str(e)}</div>',
                    unsafe_allow_html=True,
                )
                if enable_debug:
                    st.exception(e)

    # Analysis History
    if st.session_state.analysis_history:
        st.markdown("---")
        st.markdown(
            '<div class="section-header">Recent Analyses</div>', unsafe_allow_html=True
        )
