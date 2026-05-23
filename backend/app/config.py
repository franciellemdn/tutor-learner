import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables relative to this file's parent's parent (backend/.env)
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# Export sanitized constants
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip().strip("'\"")
OPENROUTER_API_BASE = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1").strip().strip("'\"")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3").strip().strip("'\"")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash").strip().strip("'\"")
