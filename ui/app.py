"""
Main entry point: `streamlit run ui/app.py`. Thin - page config, sidebar
toggles, and section ordering only. All logic lives in the other ui/ modules.
API key is loaded from the environment (.env locally, platform secrets in
deployment) - never entered by the user.
"""
import os
import sys
import uuid
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import bootstrap  # noqa: F401 - registers all built-in processes
from ui.state import init_session_state
from ui.dashboard import render_upload_section, render_target_section, render_process_checklist
from ui.chat import render_chat_section

st.set_page_config(page_title="EDA Agent V4", layout="wide")
init_session_state()

if st.session_state.session_id is None:
    st.session_state.session_id = str(uuid.uuid4())

st.title("EDA Agent V4")

with st.sidebar:
    st.session_state.enable_sandbox = st.checkbox(
        "Allow agent to run generated code (sandboxed)",
        value=st.session_state.get("enable_sandbox", False),
    )
    st.session_state.enable_hitl = st.checkbox(
        "Require approval before each tool call (HITL)",
        value=st.session_state.get("enable_hitl", False),
        help="Agent will pause and show you the proposed tool call before executing it.",
    )

render_upload_section()

if st.session_state.dco is not None:
    left, right = st.columns([3, 2])
    with left:
        render_target_section()
        render_process_checklist()
    with right:
        render_chat_section()
else:
    st.info("Upload a CSV or Parquet file to get started.")
