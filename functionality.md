# 🧠 Grasper AI — Module Functionalities

This document explains the purpose, inputs, outputs, and interconnections of all core files in the Grasper AI system.

---

## 📁 `app/main.py`

**Purpose**: Initializes FastAPI app, sets up CORS, and launches the server.

- **Input**: None directly (mounted by Uvicorn or run manually)
- **Output**: Serves the API endpoints

---

## 📁 `app/routes.py`

**Purpose**: Defines HTTP API routes

- **Input**: JSON POST requests (e.g. `data_analyst_input.txt`, file metadata)
- **Output**: Triggers the pipeline and returns final JSON answers

---

## 📁 `app/config.py`

**Purpose**: Stores configurable paths, keys, constants

- **Input**: None (read-only constants)
- **Output**: Used by other modules

---

## 📁 `app/models/schemas.py`

**Purpose**: Defines Pydantic models for request/response validation

- **Input**: JSON from routes
- **Output**: Validated and structured Python objects

---

## 📁 `app/models/prompts.py`

**Purpose**: Stores LLM prompt templates

- **Input**: Called with data_analyst_input or context
- **Output**: Ready-to-use prompt strings

---


---

## 📁 `app/agent/data_agent.py`

**Purpose**: Asks LLM to generate code for downloading data sources

- **Input**: `data_analyst_input.txt`,attached files,path text
- **Output**: JSON containing:
  - source name
  - scraping code
  - verify code
  - optional url/html/media flags

---

## 📁 `app/agent/executor.py`

**Purpose**: Manages async code execution and file handling

- **Input**: Scraping/verify code blocks, file paths
- **Output**: Downloaded files in unique `../tmp/tempX/` folders

---

## 📁 `app/agent/html_handler.py`

**Purpose**: Scrapes and parses structured data from websites

- **Input**: `url` field from data_agent output
- **Output**: Structured JSON

---

## 📁 `app/agent/media_handler.py`

**Purpose**: Processes media files (image/audio/video)

- **Input**: Media files
- **Output**: Metadata (duration, dimensions, etc)

---

## 📁 `app/agent/analyzer.py`

**Purpose**: Asks LLM to summarize file structure + headers

- **Input**: Data files
- **Output**: JSON: size, structure, head of each file

---

## 📁 `app/agent/answer_agent.py`

**Purpose**: Asks LLM to answer questions using structured data

- **Input**: data_analyst_input + structure reports + downloaded files
- **Output**: Code that answers all data_analyst_input

---

## 📁 `app/agent/validator.py`

**Purpose**: Validates LLM-generated answers

- **Input**: Code + Output
- **Output**: Boolean success + formatted JSON

---

## 📁 `app/agent/cache.py`

**Purpose**: Hash-based or Redis-like caching

- **Input**: Job hash, data_analyst_input, or files
- **Output**: Reuse of previous results if match

---

## 📁 `app/agent/utils.py`

**Purpose**: Common helper functions (logging, temp cleanup, etc)

- **Input/Output**: Varies by helper

---

## 🧪 `tests/`

Contains unit tests for agent, executor, and validator logic

---

## ✅ Overall Workflow

1. Receive `data_analyst_input.txt` + files via API
2. Parse and create temp folder for job
3. Ask LLM for data scraping plan
4. Run generated code, download files
5. If media/html, process accordingly
6. Analyze data structure
7. Ask LLM to answer questions using data
8. Validate results with LLM x 2 + Pydantic
9. Return validated structured response