# Grasper AI Data Analysis System - Complete Functional Flow

## Overview

Grasper is an AI-powered data analysis system that processes user questions, downloads/analyzes data from various sources, and generates Python code to answer those questions. The system is built using FastAPI, PydanticAI with Google Gemini 2.5 Flash, and includes comprehensive error recovery and monitoring.

## 1. API Request Layer

### Primary Endpoint

#### POST `/api/`
- **Method**: POST
- **Content-Type**: multipart/form-data
- **Purpose**: Main data analysis endpoint that processes questions and attachments
- **Authentication**: None required
- **Parameters**:
  - **Form Fields**:
    - `questions.txt` (required): File containing the data analyst input/questions
    - Any number of additional file attachments (optional): data files, images, documents
- **Headers**: Standard multipart form headers
- **Response Format**: JSON with analysis results or error information

#### GET `/health`
- **Method**: GET
- **Purpose**: Health check endpoint for system monitoring
- **Parameters**: None
- **Response**: `{"status": "ok", "message": "Grasper AI backend is live."}`

### API Request Processing Flow

The main API endpoint (`/api/`) in `app/routes.py` handles incoming requests as follows:

1. **Request Parsing**: Extracts multipart form data from the request
2. **File Validation**: 
   - Searches for `questions.txt` file (either by field name or filename)
   - Collects all other uploaded files as attachments
   - Validates that `questions.txt` is present (returns 400 error if missing)
3. **Content Reading**: Decodes `questions.txt` content as UTF-8
4. **Payload Construction**: Creates payload dictionary with:
   - `data_analyst_input`: The decoded text from questions.txt
   - `attachments`: List of uploaded file objects
5. **Delegation**: Passes payload to `handle_payload()` function

## 2. Activation Flow and Core Processing Pipeline

### Primary Handler: `handle_payload()` in `app/payload_handler.py`

This is the main orchestrator that coordinates the entire analysis workflow:

**Parameters**:
- `payload`: Dictionary containing `data_analyst_input` (string) and `attachments` (list of file objects)

**Processing Steps**:

#### Step 1: Temporary Directory Management
- **Function**: `create_temp_dir()` from `app/agent/executor.py`
- **Trigger**: Immediately upon payload receipt
- **Purpose**: Creates a unique temporary directory based on SHA256 hash of input text
- **Location**: `/tmp/{hash_prefix}/` where hash_prefix is first 10 chars of SHA256
- **Dependencies**: Uses `pathlib.Path` and `hashlib`
- **Persistence**: Directory is reused if it already exists (supports caching)

#### Step 2: File Upload Processing
- **Function**: `save_multiple_uploaded_files()` from `app/agent/executor.py`
- **Trigger**: If attachments are present in payload
- **Parameters**: 
  - `uploaded_files`: List of FastAPI UploadFile objects
  - `dest_dir`: Path object pointing to temporary directory
- **Process**:
  - Iterates through each uploaded file
  - Saves each file to temp directory using original filename
  - Uses `aiofiles` for async file I/O
  - Returns list of saved file paths
- **Error Handling**: Logs errors but continues processing
- **Dependencies**: `aiofiles`, `asyncio`

#### Step 3: Data Agent Processing
- **Function**: `handle_data()` from `app/agent/data_agent.py`
- **Trigger**: Always executed after file handling
- **Purpose**: Downloads additional data from URLs found in the input text

##### Data Agent Internal Flow:

**URL Detection and Processing**:
1. **URL Extraction**: Uses regex to find HTTP(S) URLs in input text
2. **URL Classification**: Determines if URLs point to:
   - **HTML pages**: Based on extension (.html, .htm) or Content-Type headers
   - **Data files**: CSV, JSON, Excel, Parquet, etc.
3. **Text Preprocessing**: Replaces webpage URLs with domain.path format to avoid confusion

**PydanticAI Agent Invocation**:
- **Model**: Google Gemini 2.5 Flash via PydanticAI
- **Agent Type**: `Agent` with structured output schema
- **Input Schema**: `AgentDeps` dataclass containing:
  - `data_analyst_input`: Preprocessed input text
  - `downloaded_files`: List of already saved files
  - `temp_dir`: Temporary directory path
- **Output Schema**: `DataPlan` with source mapping
- **Caching**: Plans are cached by input hash to avoid regeneration

**Execution Phase**:
For each data source identified by the agent:

1. **HTML Sources**:
   - **Function**: `scrape_html()` from `app/agent/html_handler.py`
   - **Process**: 
     - First attempts simple requests.get()
     - Falls back to Playwright for JavaScript-heavy sites
     - Saves prettified HTML to temp directory
   - **Dependencies**: `requests`, `BeautifulSoup`, optional `playwright`

2. **Data File Sources**:
   - **Security Validation**: 
     - AST parsing to validate generated Python code
     - Blocks dangerous imports (subprocess, eval, etc.)
     - Ensures only `requests.get()` is used
   - **Code Execution**:
     - Writes generated Python script to temp directory
     - Executes using subprocess with timeout
     - Handles missing dependencies via automatic pip install
   - **File Validation**: 
     - Validates downloaded files can be read (pandas, json libs)
     - Ensures files are at least 1KB in size
     - Removes generated Python scripts after validation

**Error Recovery**:
- **Model Retry**: Uses `ModelRetry` for execution failures
- **Dependency Installation**: Automatically installs missing Python packages
- **Fallback Handling**: Continues processing even if some sources fail

#### Step 4: File Analysis
- **Function**: `analyze_dir()` from `app/agent/analyzer.py`
- **Trigger**: After all data collection is complete
- **Purpose**: Analyzes all files in the temporary directory to create a comprehensive data report

##### File Analysis Internal Flow:

**File Discovery**:
1. **File Scanning**: Recursively scans temp directory for data files
2. **Archive Extraction**: Automatically extracts ZIP, TAR files to temporary subdirectories
3. **File Categorization**:
   - **Data files**: CSV, TSV, JSON, Parquet, Excel, Feather
   - **Media files**: Images, audio, video
   - **HTML files**: Web pages
   - **Archive files**: ZIP, TAR variants

**Caching System**:
- **Cache File**: `analysis_cache.json` in temp directory
- **Cache Key**: Combination of file path, modification time, and size
- **Cache Content**: File metadata and analysis summaries
- **Atomic Updates**: Uses atomic file operations to prevent corruption

**File Type Processing**:

1. **Data Files** (CSV, JSON, Excel, etc.):
   - **Sample Reading**: Reads first 10 rows for schema inference
   - **DuckDB Acceleration**: Optional fast row counting for large CSV files
   - **Schema Detection**: Infers column types, missing values, statistics
   - **Preview Generation**: Creates sample data previews
   - **Dependencies**: `pandas`, `numpy`, optional `duckdb`, `pyarrow`

2. **Media Files** (Images, Audio, Video):
   - **Function**: `analyze_media()` from `app/agent/media_analyzer.py`
   - **AI Processing**: Uses Google Gemini 2.5 Flash for content analysis
   - **Structured Output**: Extracts metadata, descriptions, detected objects
   - **Async Processing**: Processes multiple files concurrently
   - **Dependencies**: `pydantic_ai`, `mimetypes`

3. **HTML Files**:
   - **Function**: `analyze_html()` from `app/agent/html_analyzer.py`
   - **Content Extraction**: Extracts tables, lists, text content
   - **Structure Analysis**: Identifies HTML structure and data patterns
   - **Dependencies**: `BeautifulSoup`

**Report Generation**:
- **Cache Update**: Atomically updates analysis cache
- **Human-Readable Report**: Generates `analysis_report.txt` with:
  - File listings with metadata
  - Schema summaries
  - Sample data previews
  - Error reports
- **Structured Data**: Returns dictionary with complete analysis results

#### Step 5: Answer Generation
- **Function**: `answer_questions()` from `app/agent/answer_agent.py`
- **Trigger**: After file analysis is complete
- **Purpose**: Generates and executes Python code to answer the original questions

##### Answer Agent Internal Flow:

**Context Preparation**:
1. **Analysis Report Loading**: Reads `analysis_report.txt` from temp directory
2. **Context Assembly**: Combines:
   - Original data analyst input (questions)
   - File analysis report (data schemas and samples)
   - Temporary directory path for data access

**PydanticAI Agent Configuration**:
- **Model**: Google Gemini 2.5 Flash
- **Agent Type**: Structured output agent
- **Input Dependencies**: `AnswerDeps` dataclass with input text, temp dir, analysis report
- **Output Schema**: JSON object with `questions` array and `data_analysis_pythonic_code` string
- **System Prompt**: Instructs agent to write executable Python code for data analysis

**Code Generation and Execution Loop**:

1. **Initial Generation**:
   - **Prompt Construction**: Detailed prompt with data context and requirements
   - **Code Generation**: Agent produces Python script as string
   - **Validation**: Ensures output matches expected JSON schema

2. **Code Execution**:
   - **File Writing**: Saves generated code to `{temp_dir}/agent_analysis/answer_analysis.py`
   - **Subprocess Execution**: Runs Python script with 120-second timeout
   - **Output Capture**: Captures stdout, stderr, and return code
   - **Working Directory**: Executes from temp directory for data access

3. **Success Validation**:
   - **Return Code Check**: Ensures script executed without errors
   - **JSON Parsing**: Validates that stdout contains valid JSON
   - **Result Return**: Returns parsed JSON as final response

4. **Error Recovery Loop** (up to 5 attempts):
   - **Error Analysis**: Captures execution errors and stderr
   - **Retry Prompt**: Includes previous error in next generation attempt
   - **Progressive Refinement**: Each attempt includes context from previous failures
   - **Timeout Handling**: Kills runaway processes and reports timeout errors

**Output Processing**:
- **Type Conversion**: Uses `convert_numpy_types()` to ensure JSON serialization
- **Error Reporting**: Detailed error information if all attempts fail
- **Success Response**: Returns the final JSON output from successful execution

## 3. Data Handling Pipeline

### Input Data Validation and Transformation

**Request Level**:
- **Multipart Form Validation**: Ensures `questions.txt` is present
- **File Type Detection**: Uses filename and content sniffing
- **Encoding Handling**: UTF-8 with fallback error handling

**Data Agent Level**:
- **URL Validation**: Regex and HTTP header validation
- **Security Checks**: AST parsing of generated code
- **File Size Validation**: Minimum 1KB requirement for downloaded files
- **Content Validation**: Pandas/JSON library validation

**File Analysis Level**:
- **Schema Inference**: Automatic detection of data types and structure
- **Missing Value Detection**: Comprehensive null/NaN analysis
- **Statistical Summary**: Basic statistics for numeric columns
- **Sample Generation**: Representative data samples for AI context

### Data Storage and Retrieval

**Temporary Directory Structure**:
```
/tmp/{hash}/
├── analysis_cache.json          # File analysis cache
├── analysis_report.txt          # Human-readable analysis summary
├── {original_uploaded_files}    # User-provided files
├── {downloaded_data_files}      # Files downloaded from URLs
├── html/                        # Scraped HTML content
│   └── {domain_path}.html
└── agent_analysis/             # Generated analysis code
    └── answer_analysis.py
```

**Database Operations**: None - all data is file-based for simplicity and portability

**Caching Strategy**:
- **File-level caching**: Based on file metadata (path, mtime, size)
- **Plan caching**: LLM generation results cached by input hash
- **Atomic updates**: Prevents cache corruption during concurrent access

### External API Integration

**Google Gemini 2.5 Flash**:
- **Authentication**: API key via environment variable `GEMINI_API_KEY`
- **Rate Limiting**: Handled by PydanticAI library
- **Error Handling**: Automatic retries with exponential backoff
- **Token Management**: Managed by PydanticAI provider

**Web Scraping**:
- **Primary Method**: requests library with timeout
- **Fallback Method**: Playwright for JavaScript-heavy sites
- **User Agent**: Standard browser user agent
- **Timeout Configuration**: 30 seconds for requests, 2 seconds additional wait for Playwright

## 4. Processing Paths and Branching Logic

### Conditional Execution Paths

#### Data Source Detection Branch:
```
Input Text Analysis
├── No URLs Found → Skip data agent, use only uploaded files
└── URLs Found → Execute data agent workflow
    ├── HTML URLs → Use html_tool (scraping)
    └── Data URLs → Use code_tool (download)
```

#### File Type Processing Branch:
```
File Analysis
├── Data Files (.csv, .json, .xlsx, etc.)
│   ├── DuckDB Available → Fast row counting
│   └── DuckDB Unavailable → Pandas-only analysis
├── Media Files (.jpg, .mp4, .mp3, etc.)
│   └── Google Gemini Vision/Audio Analysis
├── HTML Files (.html, .htm)
│   └── BeautifulSoup structure extraction
└── Archive Files (.zip, .tar, etc.)
    └── Extract and recursively analyze contents
```

#### Code Execution Branch:
```
Answer Generation
├── First Attempt → Generate code → Execute
├── Execution Success → Parse JSON → Return result
└── Execution Failure → Include error in next prompt → Retry (up to 5 times)
    ├── Timeout Error → Kill process → Retry with timeout context
    ├── Module Missing → Auto pip install → Retry
    └── Other Error → Include stderr in retry prompt
```

### Asynchronous and Parallel Processing

**File Upload Processing**:
- **Parallel Uploads**: Multiple files uploaded concurrently using `asyncio.gather()`
- **Per-file Async I/O**: Each file written using `aiofiles`

**Data Source Processing**:
- **Sequential by Design**: Data sources processed one at a time to avoid rate limiting
- **Internal Async**: Each source uses async I/O for network operations

**Media Analysis**:
- **Parallel Processing**: Multiple media files analyzed concurrently
- **Batch Processing**: Uses `asyncio.gather()` for concurrent API calls

**File Analysis**:
- **Sequential File Processing**: Files analyzed one at a time for memory efficiency
- **Async Cache I/O**: Cache reads/writes use async file operations

### Background Jobs and Retries

**Background Processes**: None - all processing is request-scoped

**Retry Mechanisms**:
1. **PydanticAI Agent Retries**: Up to 5 retries for LLM generation failures
2. **Code Execution Retries**: Up to 5 attempts for generated code execution
3. **HTTP Request Retries**: Built into requests library with timeout
4. **Playwright Fallback**: Automatic fallback for failed requests scraping

**Error Recovery Strategies**:
- **Missing Dependencies**: Automatic pip installation during code execution
- **Network Timeouts**: Progressive timeout increases and fallback methods
- **File Access Errors**: Graceful degradation with error reporting
- **LLM Generation Failures**: Context-aware retry with error feedback

## 5. Final Response Format

 as user asked 

 
### Response Processing Pipeline

1. **Type Conversion**: All NumPy and Pandas types converted to native Python types via `convert_numpy_types()`
2. **JSON Serialization**: Ensures all response data is JSON-serializable
3. **Error Sanitization**: Sensitive paths and internal details filtered from error messages
4. **Logging Integration**: All responses logged to Logfire for monitoring
5. **HTTP Status Codes**: 
   - 200: Success with valid results
   - 400: Invalid request (missing questions.txt, invalid encoding)
   - 500: Internal processing errors

### Monitoring and Observability

**Logfire Integration**:
- **Request Tracing**: Full request lifecycle tracking
- **Performance Metrics**: Execution times for each processing stage
- **Error Tracking**: Detailed error capture with context
- **AI Analytics**: Token usage, model performance, success rates

**Metrics Exposed**:
- Request rates and response times
- Success/failure ratios
- File processing statistics
- LLM generation performance
- Error categorization

**Debug Information**:
- Temporary directory contents preserved for debugging
- Generated code saved for inspection
- Detailed execution logs with timestamps
- Agent conversation history

## 6. Dependencies and External Services

### Required Dependencies
- **FastAPI**: Web framework and request handling
- **PydanticAI**: LLM integration and structured outputs
- **Google Gemini API**: AI model for code generation and media analysis
- **Pandas/NumPy**: Data analysis and manipulation
- **BeautifulSoup**: HTML parsing and extraction
- **Logfire**: Monitoring and observability

### Optional Dependencies
- **DuckDB**: Fast analytics for large CSV files
- **PyArrow**: Efficient Parquet file handling
- **Playwright**: JavaScript-capable web scraping
- **Various data format libraries**: openpyxl, xlrd, etc.

### External Services
- **Google AI Studio API**: Gemini model access
- **Logfire Service**: Monitoring and analytics (optional)

## 6. Debugging and Inconsistency Fixes

### Fixed Inconsistencies

#### Analysis Report File Naming
**Issue Found**: The codebase had inconsistent references to analysis report files:
- `analysis_report.txt` (actual file created)
- `analysis_summary.json` (legacy/incorrect references)

**Fix Applied**: 
- Updated all file skip conditions to include both names for backward compatibility
- Added comprehensive debugging to track `analysis_report.txt` creation and usage
- Ensured consistent naming throughout the pipeline

#### File Processing Skip Logic
**Issue Found**: System files weren't consistently excluded from analysis
**Fix Applied**: 
- Updated `AVOID_FILES` to include: `{'.py', 'analysis_summary.json', 'analysis_report.txt', 'analysis_cache.json'}`
- Added debugging logs for all skipped files

### Comprehensive Debugging Added

#### File Analysis Debugging (`analyzer.py`)
```python
# Added extensive debugging throughout:
- File discovery and categorization logging
- Cache hit/miss tracking  
- Data processing step-by-step logs
- Report generation verification
- Atomic file write confirmation
```

#### Answer Generation Debugging (`answer_agent.py`)
```python
# Added detailed execution tracing:
- Analysis report reading verification
- Code generation attempt tracking
- Subprocess execution monitoring
- JSON parsing validation
- Error propagation with context
```

#### Payload Handler Debugging (`payload_handler.py`)
```python
# Added pipeline flow tracking:
- Temp directory creation verification
- File upload confirmation
- Analysis report creation validation
- Answer generation monitoring
- Error context preservation
```
