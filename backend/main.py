import os
from typing import TypedDict, List, Literal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langgraph.graph import StateGraph, END

from pathlib import Path

# Load environment variables relative to this file with override=True
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip().strip("'\"")
OPENROUTER_API_BASE = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1").strip().strip("'\"")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3").strip().strip("'\"")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash").strip().strip("'\"")

app = FastAPI(title="Tutor-Learner Agent Debate API")

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. State Definition
class ChatMessage(TypedDict):
    sender: Literal["tutor", "learner"]
    content: str
    model_used: str

class DiscussionState(TypedDict):
    topic: str
    mode: Literal["local", "cloud"]
    messages: List[ChatMessage]
    turn_count: int

# Pydantic input schema for FastAPI endpoint
class DiscussRequest(BaseModel):
    topic: str
    mode: Literal["local", "cloud"]

# Helper to get the correct LLM model client
def get_llm(mode: str, role: str):
    if mode == "cloud":
        if not OPENROUTER_API_KEY or OPENROUTER_API_KEY.startswith("your_"):
            raise HTTPException(
                status_code=400,
                detail="OpenRouter API key is missing. Please set OPENROUTER_API_KEY in the backend/.env file."
            )
        
        # Explicitly configure environment variables as fallback for the client
        os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
        os.environ["OPENAI_BASE_URL"] = OPENROUTER_API_BASE
        
        return ChatOpenAI(
            api_key=OPENROUTER_API_KEY,
            openai_api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_API_BASE,
            model=OPENROUTER_MODEL,
            temperature=0.7,
            default_headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "Tutor Learner Learning Project"
            }
        ), OPENROUTER_MODEL
    else:
        # Local Ollama configuration
        try:
            return ChatOllama(
                model=OLLAMA_MODEL,
                temperature=0.7,
                base_url="http://localhost:11434"
            ), OLLAMA_MODEL
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to connect to local Ollama server. Ensure Ollama is running and '{OLLAMA_MODEL}' model is pulled. Error: {str(e)}"
            )

# 2. Node Implementations
def tutor_node(state: DiscussionState) -> DiscussionState:
    messages = state["messages"]
    topic = state["topic"]
    mode = state["mode"]
    
    # Get LLM based on user selection
    llm, model_name = get_llm(mode, "tutor")
    
    # Construct conversation history for the prompt
    history_str = ""
    for msg in messages:
        sender_name = "Tutor" if msg["sender"] == "tutor" else "Learner"
        history_str += f"{sender_name}: {msg['content']}\n"
    
    # Tutor System Prompt
    system_prompt = (
        f"You are a helpful, patient, and knowledgeable AI Tutor teaching a student about '{topic}'.\n"
        "Your goal is to explain concepts clearly, keep explanations brief (under 3 sentences), "
        "and ask simple questions or pose quick quizzes to check the learner's understanding.\n"
        "Always stay in character as the Tutor. Do not say things like 'Sure, here is the response:'."
    )
    
    user_prompt = (
        f"Conversation history so far:\n{history_str}\n"
        "Generate the next response as the Tutor. If the history is empty, introduce the topic and ask the learner "
        "what they already know about it or what specific questions they have."
    )
    
    # Call Model
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        content = response.content.strip()
    except Exception as e:
        content = f"[Tutor Error: Could not generate response. Details: {str(e)}]"
        model_name = "Error Node"
        
    new_message: ChatMessage = {
        "sender": "tutor",
        "content": content,
        "model_used": model_name
    }
    
    return {
        **state,
        "messages": messages + [new_message],
        "turn_count": state["turn_count"] + 1
    }

def learner_node(state: DiscussionState) -> DiscussionState:
    messages = state["messages"]
    topic = state["topic"]
    mode = state["mode"]
    
    # Get LLM based on user selection
    llm, model_name = get_llm(mode, "learner")
    
    # Construct conversation history for the prompt
    history_str = ""
    for msg in messages:
        sender_name = "Tutor" if msg["sender"] == "tutor" else "Learner"
        history_str += f"{sender_name}: {msg['content']}\n"
        
    # Learner System Prompt
    system_prompt = (
        f"You are a curious and polite AI Learner who is actively learning about '{topic}' from a Tutor.\n"
        "Your goal is to ask insightful questions, try your best to answer the tutor's quizzes/questions, "
        "and show what you understand. Keep your responses short (under 3 sentences).\n"
        "Always stay in character as the Learner. Do not say things like 'Sure, here is the response:'."
    )
    
    user_prompt = (
        f"Conversation history so far:\n{history_str}\n"
        "Generate the next response as the Learner."
    )
    
    # Call Model
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        content = response.content.strip()
    except Exception as e:
        content = f"[Learner Error: Could not generate response. Details: {str(e)}]"
        model_name = "Error Node"
        
    new_message: ChatMessage = {
        "sender": "learner",
        "content": content,
        "model_used": model_name
    }
    
    return {
        **state,
        "messages": messages + [new_message],
        "turn_count": state["turn_count"] + 1
    }

# 3. Router logic
def router_condition(state: DiscussionState) -> str:
    # Stop discussion after 6 total turns (3 rounds of tutor-learner interaction)
    if state["turn_count"] >= 6:
        return "end"
    
    # Otherwise, alternate turns based on who spoke last
    last_sender = state["messages"][-1]["sender"]
    if last_sender == "tutor":
        return "learner"
    else:
        return "tutor"

# 4. Compile LangGraph Workflow
workflow = StateGraph(DiscussionState)

# Add nodes to graph
workflow.add_node("tutor", tutor_node)
workflow.add_node("learner", learner_node)

# Set the starting node
workflow.set_entry_point("tutor")

# Define conditional edges to loop/alternate
workflow.add_conditional_edges(
    "tutor",
    router_condition,
    {
        "learner": "learner",
        "end": END
    }
)
workflow.add_conditional_edges(
    "learner",
    router_condition,
    {
        "tutor": "tutor",
        "end": END
    }
)

discussion_graph = workflow.compile()

# 5. API Endpoints
@app.post("/discuss")
async def start_debate(request: DiscussRequest):
    # Initialize state
    initial_state: DiscussionState = {
        "topic": request.topic,
        "mode": request.mode,
        "messages": [],
        "turn_count": 0
    }
    
    try:
        # Run graph synchronously
        final_state = discussion_graph.invoke(initial_state)
        return {
            "topic": final_state["topic"],
            "mode": final_state["mode"],
            "messages": final_state["messages"],
            "turn_count": final_state["turn_count"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph execution failed: {str(e)}")

@app.get("/health")
async def health_check():
    # Helper to check if Ollama is accessible and which models are pulled,
    # and if env has OpenRouter API Key.
    ollama_ok = False
    openrouter_configured = bool(OPENROUTER_API_KEY and not OPENROUTER_API_KEY.startswith("your_"))
    
    import urllib.request
    import json
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as response:
            if response.status == 200:
                ollama_ok = True
    except Exception:
        pass
        
    return {
        "status": "healthy",
        "ollama_connected": ollama_ok,
        "ollama_model": OLLAMA_MODEL,
        "openrouter_configured": openrouter_configured,
        "openrouter_model": OPENROUTER_MODEL
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
