import sqlite3
import os
from typing import Dict, Any

DB_DIR = "evaluation"
DB_PATH = os.path.join(DB_DIR, "eval_results.db")

def init_db(db_path: str = DB_PATH):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS evaluations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trace_id TEXT,
        model_name TEXT,
        sample_id TEXT,
        query TEXT,
        context TEXT,
        faithfulness INTEGER,
        relevance INTEGER,
        latency REAL,
        created_at INTEGER
    )
    """)
    conn.commit()
    conn.close()

def save_evaluation(record: Dict[str, Any], db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
    INSERT INTO evaluations (
        trace_id, model_name, sample_id, query, context,
        faithfulness, relevance, latency, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        record.get("trace_id"),
        record.get("model_name"),
        record.get("sample_id"),
        record.get("query"),
        record.get("context"),
        int(record.get("faithfulness", 0)),
        int(record.get("relevance", 0)),
        float(record.get("latency", 0.0)),
        int(record.get("created_at")),
    ))
    conn.commit()
    conn.close()
