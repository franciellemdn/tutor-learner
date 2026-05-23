import json
import sqlite3
from typing import Literal
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import configurations, database, and graph
from app.config import OLLAMA_MODEL, OPENROUTER_MODEL, OPENROUTER_API_KEY
from app.database import (
    get_all_debates,
    get_debate_messages,
    get_debate_metadata,
    create_debate,
    save_message,
    save_debate_evaluation,
    DB_PATH
)
from app.graph.state import DiscussionState
from app.graph.workflow import discussion_graph

app = FastAPI(title="Tutor-Learner Agent Debate API")

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class DiscussRequest(BaseModel):
    topic: str
    mode: Literal["local", "cloud"]
    model: str

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
        metadata = get_debate_metadata(debate_id)
        
        evaluation = None
        model = ""
        if metadata:
            model = metadata.get("model", "")
            eval_raw = metadata.get("evaluation")
            if eval_raw:
                try:
                    evaluation = json.loads(eval_raw)
                except Exception:
                    pass
                
        return {"messages": messages, "model": model, "evaluation": evaluation}
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
        model_choice = payload.get("model", "").strip()

        if not topic:
            await websocket.send_json({"error": "Topic is required"})
            await websocket.close()
            return

        # 1. Create a debate session record
        debate_id = create_debate(topic, mode, model_choice)

        initial_state: DiscussionState = {
            "topic": topic,
            "mode": mode,
            "model": model_choice,
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
