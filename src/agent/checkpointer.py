"""
SQLite-backed checkpointing for the agent graph: full message history and
tool-call trace persisted per thread_id, resumable across sessions.
"""
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from ..config import CONFIG


def get_checkpointer(db_path: str = None) -> SqliteSaver:
    db_path = CONFIG.memory.checkpoint_db_path if db_path is None else db_path
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return SqliteSaver(conn)
