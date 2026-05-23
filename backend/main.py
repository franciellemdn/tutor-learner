import os
import sqlite3
import json
from typing import TypedDict, List, Literal
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langgraph.graph import StateGraph, END

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

# ==========================================
# DATABASE SETUP (SQLite)
# ==========================================
DB_PATH = os.path.join(os.path.dirname(__file__), "debates.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Create debates table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS debates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL,
            mode TEXT NOT NULL,
            evaluation TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Database migration: add evaluation column if it doesn't exist
    try:
        cursor.execute("ALTER TABLE debates ADD COLUMN evaluation TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists
        
    # Create messages table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            debate_id INTEGER,
            sender TEXT NOT NULL,
            content TEXT NOT NULL,
            model_used TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(debate_id) REFERENCES debates(id)
        )
    """)
    conn.commit()
    conn.close()

# Initialize DB on load
init_db()

def create_debate(topic: str, mode: str) -> int:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO debates (topic, mode) VALUES (?, ?)",
        (topic, mode)
    )
    debate_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return debate_id

def save_message(debate_id: int, sender: str, content: str, model_used: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO messages (debate_id, sender, content, model_used) VALUES (?, ?, ?, ?)",
        (debate_id, sender, content, model_used)
    )
    conn.commit()
    conn.close()

def save_debate_evaluation(debate_id: int, evaluation_json: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE debates SET evaluation = ? WHERE id = ?",
        (evaluation_json, debate_id)
    )
    conn.commit()
    conn.close()

def get_all_debates():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT id, topic, mode, created_at FROM debates ORDER BY created_at DESC")
    rows = cursor.fetchall()
    debates = [dict(row) for row in rows]
    conn.close()
    return debates

def get_debate_messages(debate_id: int):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT sender, content, model_used, created_at FROM messages WHERE debate_id = ? ORDER BY id ASC",
        (debate_id,)
    )
    rows = cursor.fetchall()
    messages = [dict(row) for row in rows]
    conn.close()
    return messages

# ==========================================
# LANGGRAPH GRAPH DEFINITION
# ==========================================

class ChatMessage(TypedDict):
    sender: Literal["tutor", "learner"]
    content: str
    model_used: str

class DiscussionState(TypedDict):
    topic: str
    mode: Literal["local", "cloud"]
    messages: List[ChatMessage]
    turn_count: int
    evaluation: dict

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

# Node Implementations
def tutor_node(state: DiscussionState) -> DiscussionState:
    messages = state["messages"]
    topic = state["topic"]
    mode = state["mode"]
    
    llm, model_name = get_llm(mode, "tutor")
    
    history_str = ""
    for msg in messages:
        sender_name = "Tutor" if msg["sender"] == "tutor" else "Learner"
        history_str += f"{sender_name}: {msg['content']}\n"
    
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
    
    llm, model_name = get_llm(mode, "learner")
    
    history_str = ""
    for msg in messages:
        sender_name = "Tutor" if msg["sender"] == "tutor" else "Learner"
        history_str += f"{sender_name}: {msg['content']}\n"
        
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

def evaluator_node(state: DiscussionState) -> DiscussionState:
    messages = state["messages"]
    topic = state["topic"]
    mode = state["mode"]
    
    llm, model_name = get_llm(mode, "evaluator")
    
    history_str = ""
    for msg in messages:
        sender_name = "Tutor" if msg["sender"] == "tutor" else "Learner"
        history_str += f"{sender_name}: {msg['content']}\n"
        
    system_prompt = (
        "You are an objective AI Tutor Evaluator. Your job is to analyze the debate/conversation between the Tutor and the Learner "
        f"about the topic '{topic}' and grade the Learner's performance and conceptual understanding.\n"
        "You MUST respond ONLY with a raw JSON object (no markdown block, no ```json, no extra text). "
        "The JSON object must follow this schema exactly:\n"
        "{\n"
        "  \"score\": <integer between 1 and 10>,\n"
        "  \"summary\": \"<2-3 sentence overall critique of the learner's understanding>\",\n"
        "  \"strengths\": [\"<strength 1>\", \"<strength 2>\"],\n"
        "  \"gaps\": [\"<concept misunderstood or missed 1>\", \"<concept misunderstood 2>\"],\n"
        "  \"recommendations\": [\"<actionable reading or advice 1>\", \"<actionable reading or advice 2>\"]\n"
        "}"
    )
    
    user_prompt = (
        f"Here is the dialogue transcript:\n{history_str}\n"
        "Please evaluate the Learner and return the raw JSON scorecard."
    )
    
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        content = response.content.strip()
        
        # Clean potential markdown wrapping
        if content.startswith("```"):
            lines = content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()
            
        evaluation_data = json.loads(content)
    except Exception as e:
        evaluation_data = {
            "score": 0,
            "summary": f"Could not perform evaluation. Error: {str(e)}",
            "strengths": ["Evaluation failed"],
            "gaps": ["Error parsing response"],
            "recommendations": ["Ensure API keys are configured and local models are pulled"]
        }
        
    return {
        **state,
        "evaluation": evaluation_data
    }

# Router logic
def router_condition(state: DiscussionState) -> str:
    # Stop debate and send to evaluator after 6 turns
    if state["turn_count"] >= 6:
        return "evaluator"
    
    last_sender = state["messages"][-1]["sender"]
    if last_sender == "tutor":
        return "learner"
    else:
        return "tutor"

# Compile LangGraph Workflow
workflow = StateGraph(DiscussionState)

workflow.add_node("tutor", tutor_node)
workflow.add_node("learner", learner_node)
workflow.add_node("evaluator", evaluator_node)

workflow.set_entry_point("tutor")

workflow.add_conditional_edges(
    "tutor",
    router_condition,
    {
        "learner": "learner",
        "evaluator": "evaluator"
    }
)
workflow.add_conditional_edges(
    "learner",
    router_condition,
    {
        "tutor": "tutor",
        "evaluator": "evaluator"
    }
)
workflow.add_edge("evaluator", END)

discussion_graph = workflow.compile()

# ==========================================
# API ENDPOINTS & WEBSOCKETS
# ==========================================

@app.get("/debates")
async def list_debates():
    try:
        return get_all_debates()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/debates/{debate_id}")
async def get_debate(debate_id: int):
    try:
        messages = get_debate_messages(debate_id)
        
        # Fetch the evaluation metadata
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT evaluation FROM debates WHERE id = ?", (debate_id,))
        row = cursor.fetchone()
        conn.close()
        
        evaluation = None
        if row and row["evaluation"]:
            try:
                evaluation = json.loads(row["evaluation"])
            except Exception:
                pass
                
        return {"messages": messages, "evaluation": evaluation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.websocket("/ws/discuss")
async def websocket_discuss(websocket: WebSocket):
    await websocket.accept()
    try:
        # Receive parameters
        data = await websocket.receive_text()
        payload = json.loads(data)
        topic = payload.get("topic", "").strip()
        mode = payload.get("mode", "local").strip()

        if not topic:
            await websocket.send_json({"error": "Topic is required"})
            await websocket.close()
            return

        # 1. Create a debate session record
        debate_id = create_debate(topic, mode)

        initial_state: DiscussionState = {
            "topic": topic,
            "mode": mode,
            "messages": [],
            "turn_count": 0,
            "evaluation": {}
        }

        # 2. Execute graph step-by-step and stream results in real-time
        async for chunk in discussion_graph.astream(initial_state, stream_mode="updates"):
            print(f"[WS DEBUG] Chunk received: {chunk}")
            for node_name, state_update in chunk.items():
                print(f"[WS DEBUG] Processing node: '{node_name}'")
                if node_name == "evaluator":
                    # Evaluator completed!
                    eval_data = state_update.get("evaluation", {})
                    print(f"[WS DEBUG] Evaluator output data: {eval_data}")
                    
                    # Save evaluation JSON string to DB
                    save_debate_evaluation(debate_id, json.dumps(eval_data))
                    
                    # Stream evaluation data to client
                    await websocket.send_json({
                        "type": "evaluation",
                        **eval_data
                    })
                elif "messages" in state_update and state_update["messages"]:
                    last_msg = state_update["messages"][-1]
                    
                    # Save message
                    save_message(
                        debate_id=debate_id,
                        sender=last_msg["sender"],
                        content=last_msg["content"],
                        model_used=last_msg["model_used"]
                    )
                    
                    # Send message
                    await websocket.send_json({
                        "type": "message",
                        "sender": last_msg["sender"],
                        "content": last_msg["content"],
                        "model_used": last_msg["model_used"]
                    })

        # Complete notification
        await websocket.send_json({"type": "complete", "debate_id": debate_id})

    except WebSocketDisconnect:
        print("WebSocket client disconnected.")
    except Exception as e:
        try:
            await websocket.send_json({"error": f"Graph execution failed: {str(e)}"})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass

@app.get("/health")
async def health_check():
    ollama_ok = False
    openrouter_configured = bool(OPENROUTER_API_KEY and not OPENROUTER_API_KEY.startswith("your_"))
    
    import urllib.request
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
