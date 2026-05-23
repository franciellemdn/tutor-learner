# Autonomous Tutor & Learner Agent Debate with AI Scorecard

A production-grade, educational portfolio project built with **LangChain**, **LangGraph**, **FastAPI (WebSockets)**, **SQLite**, **Ollama**, and **OpenRouter**.

It simulates an autonomous discussion between a **Tutor Agent** and a **Learner Agent** on any user-selected topic, followed by an objective **AI Evaluator Agent** (LLM-as-a-Judge) that scores the session and provides detailed study recommendations. The entire workflow runs in **real-time** over WebSockets, letting you watch the conversation unfold turn-by-turn.

---

## 🌟 Key Features

- **Real-Time WebSocket Streaming**: Dialogue turns and evaluation results are streamed dynamically from the server to the browser as the graph executes.
- **SQLite Database Persistence**: Saves all debate sessions and individual chat messages in a local `debates.db` database.
- **Debate History Sidebar**: Allows users to navigate and reload any past debate and review the full conversation transcript and AI Scorecard.
- **AI Evaluator Node (LLM-as-a-Judge)**: Analyzes the complete transcript, scores the student's understanding from 1 to 10, identifies key strengths/gaps, and recommends next steps.
- **Flexible Execution Modes**:
  - 💻 **Local Mode**: Runs entirely offline using local LLMs via **Ollama**.
  - ☁️ **Cloud Mode**: Runs in the cloud using frontier LLMs via **OpenRouter**.
- **Aesthetic Light UI**: A clean, modern dashboard built with Vanilla CSS variables and reactive JavaScript.

---

## 📁 Project Structure

```text
tutor-learner/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py       # Environment variable parsing and cleaning
│   │   ├── database.py     # SQLite connection context and CRUD operations
│   │   ├── models.py       # Dynamic LangChain LLM instantiators
│   │   ├── graph/
│   │   │   ├── __init__.py
│   │   │   ├── state.py    # TypedDict schemas for discussion state
│   │   │   ├── nodes.py    # Tutor, Learner, and Evaluator agent functions
│   │   │   └── workflow.py # StateGraph registration and compilation
│   │   └── main.py         # FastAPI WebSocket and HTTP endpoints
│   ├── .env.example        # Example environment variables
│   ├── .env                # Active environment variables (gitignored)
│   ├── debates.db          # SQLite database file (generated automatically)
│   ├── requirements.txt    # Backend dependency file
│   └── uvicorn_run.py      # Entrypoint helper script to launch the server
├── frontend/
│   └── index.html          # Light-themed front-end dashboard
├── requirements.txt        # Root level copy of dependencies
└── README.md               # Project documentation
```

---

## 🚀 Setup & Execution Guide

### Prerequisite 1: Local Model Setup (Ollama)
1. Download and install [Ollama](https://ollama.com/).
2. Run Ollama on your machine.
3. Pull the default model (`llama3`):
   ```bash
   ollama pull llama3
   ```
   *(Note: You can change the model by updating the `OLLAMA_MODEL` variable in `backend/.env`)*

### Prerequisite 2: Cloud Model Setup (OpenRouter)
1. Register and get an API key at [OpenRouter](https://openrouter.ai/).
2. Duplicate `backend/.env.example` as `backend/.env`.
3. Add your key:
   ```env
   OPENROUTER_API_KEY=your_key_here
   ```

---

### Running the Project

#### Step 1: Install Dependencies
Open your terminal in the directory and ensure your python environment (e.g. your conda environment) is active, then run:
```bash
pip install -r requirements.txt
```

#### Step 2: Start the API Server
Run the launcher script:
```bash
cd backend
python uvicorn_run.py
```
The server will start on [http://localhost:8000](http://localhost:8000). You can verify its health at [http://localhost:8000/health](http://localhost:8000/health).

#### Step 3: Open the Frontend
Simply double-click or open `frontend/index.html` in any web browser.

1. Verify connection indicators (top-right status bar) show green for your active model providers.
2. Enter a topic (e.g., *French Revolution*).
3. Select your mode (Local or Cloud).
4. Click **Start Discussion** and watch the agents converse and evaluate in real-time!
5. Click on items in the **Debate History** sidebar on the left to review past sessions.

Made by Francielle Marques @franciellemdn with Antigravity2.0
