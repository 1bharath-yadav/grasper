from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logfire

from app.routes import router  # Your custom routes
from app.config import API_HOST, API_PORT

# Configure Logfire
logfire.configure()

app = FastAPI(
    title="Grasper AI: Data Analyst Agent",
    description="LLM-powered automated data analysis with Pydantic validation and caching.",
    version="1.0.0",
)

# Add Logfire middleware
logfire.instrument_fastapi(app)

# CORS setup (you can customize allowed origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount core API routes
app.include_router(router, prefix="/api")


# Optional: Health check route
@app.get("/health", tags=["System"])
def health_check():
    logfire.info("Health check requested")
    return {"status": "ok", "message": "Grasper AI backend is live."}


# Run with: python app/main.py OR uvicorn app.main:app --reload
if __name__ == "__main__":
    logfire.info("Starting Grasper AI server")
    uvicorn.run("app.main:app", host=API_HOST, port=API_PORT, reload=True)
