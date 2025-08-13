from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()
PROVIDER_URL = os.getenv("PROVIDER_URL", "http://localhost:8000/api/")
