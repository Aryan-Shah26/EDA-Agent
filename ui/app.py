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

from src import bootstrap
from ui.state import init_session_state
from ui.dashboard import render_upload_section, render_target_tab, render_category_checklist, render_data_profile_tab
from ui.chat import render_chat_section

st.set_page_config(page_title="EDA Agent V4", layout="wide")
init_session_state()

if st.session_state.session_id is None:
    st.session_state.session_id = str(uuid.uuid4())

st.title("EDA Agent V4")


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.header("Agent Settings")
    st.session_state.enable_sandbox = st.checkbox(
        "Allow agent to run generated code (sandboxed)",
        value=st.session_state.get("enable_sandbox", False),
    )
    st.session_state.enable_hitl = st.checkbox(
        "Require approval before each tool call (HITL)",
        value=st.session_state.get("enable_hitl", False),
        help="Agent will pause and show you the proposed tool call before executing it.",
    )
    st.divider()
    st.header("Data Source")
    render_upload_section()

# ==========================================
# MAIN CANVAS (The 6 Tabs)
# ==========================================
if st.session_state.dco is not None:
    tab_profile, tab_target, tab_math, tab_viz, tab_clean, tab_chat = st.tabs([
        "Data Profile", 
        "Target Analysis", 
        "EDA", 
        "Visualizations", 
        "Data Cleaning & Eng", 
        "AI Co-Pilot"
    ])
    
    with tab_profile:
        render_data_profile_tab()
        
    with tab_target:
        render_target_tab()
        
    with tab_math:
        render_category_checklist("Mathematical & Statistical Analyses", "statistic")
        
    with tab_viz:
        render_category_checklist("Visual Exploratory Data Analysis", "visualization")
        
    with tab_clean:
        # Maps to model_suggestion category (Feature Eng, Outliers, Nulls)
        render_category_checklist("Cleaning, Outliers & Feature Engineering", "model_suggestion")
        
    with tab_chat:
        render_chat_section()
else:
    st.info("Please upload a CSV or Parquet file in the sidebar to get started.")