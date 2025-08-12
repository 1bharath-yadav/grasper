# 🎯 Grasper - AI-Powered Data Analysis Agent

Grasper is a production-ready AI data analysis system built with **PydanticAI**, **Google Gemini 2.0 Flash**, and **functional programming principles**. It provides a complete workflow from questions to executable code with memory management, error recovery, and real-time monitoring.

## 🌟 Key Features

### 🤖 **Complete AI Workflow**
- **Questions → Tasks**: Breaks down complex questions into executable tasks
- **Code Generation**: Uses Google Gemini 2.0 Flash via PydanticAI to generate Python code
- **Execution**: Safely executes code using subprocess with timeout and error handling
- **Error Recovery**: Automatically detects and fixes code errors in a loop until successful
- **Memory Management**: Maintains workflow state and results throughout the process

### 🛠 **Technical Excellence**
- **Real-time Monitoring**: Logfire integration for observability and debugging
- **Concurrent Processing**: Parallel task execution with throttling and rate limiting
- **Type Safety**: Full type hints with Pydantic models
- **Error Resilience**: Comprehensive error handling and recovery mechanisms

## 🚀 Quick Start

### 1. **Setup Environment**

```bash
# Install dependencies with uv
uv sync

# Create environment file from template
cp .env.example .env
```

### 2. **Configure API Keys**

Edit `.env` file:

```bash
# Required: Google API Key for Gemini
GOOGLE_API_KEY=your-google-api-key-here

# Optional: Logfire for monitoring
LOGFIRE_TOKEN=your-logfire-token

# Optional: Environment settings
ENVIRONMENT=development
DEBUG=true
```

**Get Google API Key:**
1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Create a new API key
3. Add it to your `.env` file

### 3. **Start the Server**

```bash
# Development server with hot reload
uv run uvicorn api.complete_api:app --host 0.0.0.0 --port 8000 --reload
```

## 📋 Usage Examples

### 1. **Process Data Analyst Input File**

Add your data analyst input to `data_analyst_input.txt`, then:

```
curl "https://app.example.com/api/" -F "data_analyst_input.txt=@data_analyst_input.txt" -F "image.png=@image.png" -F "data.csv=@data.csv" 
```

## 🏗 Architecture


```
Data Analyst Input → Task Breakdown → Code Generation → Execution → Error Recovery → Results
```

**Core Components:**
- **`agents/`**: AI agents using PydanticAI with Google Gemini
- **`core/`**: Workflow engine, functional utilities, type definitions
- **`api/`**: FastAPI endpoints with real-time monitoring
- **`monitoring/`**: Logfire integration and observability

## 📊 Monitoring & Observability

- **Logfire Integration**: Real-time logs, traces, and metrics
- **Performance Metrics**: Request rates, response times, error rates
- **AI Analytics**: Token usage, model performance, success rates

Access metrics at: `http://localhost:8000/metrics`

## 99% code written by llm
---

**Built with ❤️ using PydanticAI**
