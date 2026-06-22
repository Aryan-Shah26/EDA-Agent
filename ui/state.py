"""
Centralized session_state keys/defaults. One place to see every piece of
state the UI keeps, instead of magic strings scattered across dashboard.py/
chat.py/app.py.
"""
import streamlit as st


def init_session_state():
    defaults = {
        "dco": None,
        "csv_path": None,
        "process_results": {},
        "chat_messages": [],
        "chat_memory": None,
        "compiled_graph": None,
        "enable_sandbox": False,
        "enable_hitl": False,
        "session_id": None,
        # Testing seams only - never set by real user interaction
        "_test_llm_override": None,
        "_test_fast_llm_override": None,
        "_last_hitl": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
