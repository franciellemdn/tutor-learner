import os
from fastapi import HTTPException
from langchain_openai import ChatOpenAI
try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama
from app.config import OPENROUTER_API_KEY, OPENROUTER_API_BASE, OLLAMA_MODEL, OPENROUTER_MODEL

def get_llm(mode: str, role: str, model_name: str):
    if mode == "cloud":
        if not OPENROUTER_API_KEY or OPENROUTER_API_KEY.startswith("your_"):
            raise HTTPException(
                status_code=400,
                detail="OpenRouter API key is missing. Please set OPENROUTER_API_KEY in the backend/.env file."
            )
        
        # Explicitly configure environment variables as fallback for the client
        os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
        os.environ["OPENAI_BASE_URL"] = OPENROUTER_API_BASE
        
        selected_model = model_name or OPENROUTER_MODEL
        
        return ChatOpenAI(
            api_key=OPENROUTER_API_KEY,
            openai_api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_API_BASE,
            model=selected_model,
            temperature=0.7,
            default_headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "Tutor Learner Learning Project"
            }
        ), selected_model
    else:
        # Local Ollama configuration
        selected_model = model_name or OLLAMA_MODEL
        try:
            return ChatOllama(
                model=selected_model,
                temperature=0.7,
                base_url="http://localhost:11434"
            ), selected_model
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to connect to local Ollama server. Ensure Ollama is running and '{selected_model}' model is pulled. Error: {str(e)}"
            )
