# Autonomous Tutor & Learner Agent Debate

A simple, educational, and production-ready portfolio project built with **LangChain**, **LangGraph**, **FastAPI**, **Ollama**, and **OpenRouter**. 

It simulates an autonomous discussion between a **Tutor Agent** and a **Learner Agent** on any user-selected topic. The user can choose to run this debate **entirely locally** using local models via Ollama or **in the cloud** using frontier models via OpenRouter.

---

## 🌟 Key Concepts Demonstrated

- **LangGraph Stateful Orchestration**: Implements a cyclic agent loop (Tutor 🔄 Learner) with message exchange and turn-count limits.
- **Dynamic Model Selection**: Demonstrates how to write code that swaps between local API calls (Ollama) and cloud APIs (OpenRouter) on-the-fly.
- **FastAPI Backend Integration**: Exposes the graph execution over clean HTTP endpoints. Includes an active health checking routine to inspect local/cloud model state.
- **Aesthetic Light UI**: A modern, responsive single-page dashboard built with Vanilla CSS variables and clean Javascript for real-time visualization.

---

## 📁 Project Structure

```text
tutor-learner/
├── backend/
│   ├── .env.example       # Example environment variables
│   ├── .env               # Active environment variables (gitignored)
│   ├── main.py            # FastAPI endpoints, LangGraph, and Agent nodes
│   └── requirements.txt   # Backend dependency file
├── frontend/
│   └── index.html         # Light-themed front-end dashboard
├── requirements.txt       # Root level copy of dependencies
└── README.md              # Project documentation
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
Open your terminal in the `backend/` directory, activate your virtual environment, and run:
```bash
pip install -r requirements.txt
```
*(On Windows with Anaconda, make sure your conda/DLL paths are set up if you encounter SSL import warnings).*

#### Step 2: Start the Backend Server
Run the FastAPI development server:
```bash
uvicorn main:app --reload
```
The server will start on [http://localhost:8000](http://localhost:8000). You can verify its health at [http://localhost:8000/health](http://localhost:8000/health).

#### Step 3: Open the Frontend
Simply double-click or open `frontend/index.html` in any web browser. 

1. Check the top status indicators to see if Ollama or OpenRouter is successfully connected.
2. Enter a topic (e.g., *Quantum Computing*).
3. Select your mode (Local or Cloud).
4. Click **Start Discussion** and watch the agents converse autonomously!
