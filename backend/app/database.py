import os
import sqlite3
import json

# Database path (backend/debates.db)
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "debates.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Create debates table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS debates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL,
            mode TEXT NOT NULL,
            model TEXT,
            evaluation TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Database migration checks
    try:
        cursor.execute("ALTER TABLE debates ADD COLUMN evaluation TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists
    try:
        cursor.execute("ALTER TABLE debates ADD COLUMN model TEXT")
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

def create_debate(topic: str, mode: str, model: str) -> int:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO debates (topic, mode, model) VALUES (?, ?, ?)",
        (topic, mode, model)
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
    cursor.execute("SELECT id, topic, mode, model, created_at FROM debates ORDER BY created_at DESC")
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

def get_debate_metadata(debate_id: int):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT model, evaluation FROM debates WHERE id = ?", (debate_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def delete_debate(debate_id: int):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM messages WHERE debate_id = ?", (debate_id,))
    cursor.execute("DELETE FROM debates WHERE id = ?", (debate_id,))
    conn.commit()
    conn.close()
