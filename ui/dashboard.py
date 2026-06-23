"""
Dashboard: upload -> profile -> target confirmation -> a checklist of
registered processes the user opts into, run in parallel via
tools/scheduler.py. Each result renders the moment it's ready (inside the
button-press block, via per-process placeholders) - that's the actual
point of the scheduler being a generator, not just a backend detail.
NETWORK-cost processes are left out of the checklist entirely (no search
provider is wired in this build yet); LLM-cost ones only appear once a
Groq API key is set.
"""
import os
import tempfile
import streamlit as st
import pandas as pd
import numpy as np

from src.ingestion.profiler import profile_dataset, get_class_counts
from src.target_analysis.detector import detect_target_candidates
from src.target_analysis.health_audit import audit_target_health
from src.tools.registry import REGISTRY, ProcessCost
from src.tools.scheduler import run_selected
from src.agent.llm_router import get_llm
from .render import render_result
from src.tools.builtin_processes import run_automl_suite


def render_upload_section():
    uploaded = st.file_uploader("Upload a CSV or Parquet file", type=["csv", "parquet"])
    
    if uploaded is None:
        st.session_state.current_upload_signature = None
        return

    file_signature = f"{uploaded.name}_{uploaded.size}"
    if st.session_state.get("current_upload_signature") == file_signature:
        return  

    suffix = os.path.splitext(uploaded.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        csv_path = tmp.name

    with st.spinner("Profiling dataset (DuckDB, out-of-core - safe for large files)..."):
        st.session_state.dco = profile_dataset(csv_path)
        
    st.session_state.csv_path = csv_path
    st.session_state.process_results = {}
    st.session_state.current_upload_signature = file_signature


def _get_optimization_insight(prof, sample_series):
    """
    Evaluates a column profile against strict memory bounds and formatting heuristics
    to recommend data type downcasting or datetime conversions.
    """
    dtype = prof.dtype.upper()
    
    # 1. Float Downcasting
    if dtype in ["DOUBLE", "FLOAT", "REAL"] and prof.min_val is not None and prof.max_val is not None:
        if prof.min_val >= np.finfo(np.float16).min and prof.max_val <= np.finfo(np.float16).max:
            return "Downcast to float16"
        elif prof.min_val >= np.finfo(np.float32).min and prof.max_val <= np.finfo(np.float32).max and dtype == "DOUBLE":
            return "Downcast to float32"
            
    # 2. Integer Downcasting
    elif dtype in ["BIGINT", "INTEGER", "HUGEINT", "SMALLINT", "TINYINT"] and prof.min_val is not None and prof.max_val is not None:
        if prof.min_val >= np.iinfo(np.uint8).min and prof.max_val <= np.iinfo(np.uint8).max:
            return "Downcast to uint8"
        elif prof.min_val >= np.iinfo(np.int8).min and prof.max_val <= np.iinfo(np.int8).max:
            return "Downcast to int8"
        elif prof.min_val >= np.iinfo(np.uint16).min and prof.max_val <= np.iinfo(np.uint16).max:
            return "Downcast to uint16"
        elif prof.min_val >= np.iinfo(np.int16).min and prof.max_val <= np.iinfo(np.int16).max:
            return "Downcast to int16"
        elif prof.min_val >= np.iinfo(np.uint32).min and prof.max_val <= np.iinfo(np.uint32).max:
            return "Downcast to uint32"
        elif prof.min_val >= np.iinfo(np.int32).min and prof.max_val <= np.iinfo(np.int32).max and dtype in ["BIGINT", "HUGEINT"]:
            return "Downcast to int32"
            
    # 3. Object to Datetime (Heuristic)
    elif dtype in ["VARCHAR", "TEXT", "OBJECT"] and sample_series is not None:
        s = sample_series.dropna().head(50)
        # Check if it has typical date separators to avoid falsely parsing raw numbers
        if len(s) > 0 and s.astype(str).str.contains(r'[-/:]').all():
            try:
                pd.to_datetime(s, errors='raise')
                return "Convert to Datetime"
            except Exception:
                pass
                
    return "-"

def render_data_profile_tab():
    dco = st.session_state.dco
    st.subheader("Dataset Overview")
    
    st.write(f"**Rows:** `{dco.n_rows:,}` | **Columns:** `{dco.n_cols}`")
    
    flags = dco.flags
    if flags:
        for f in flags:
            level = {"critical": st.error, "warning": st.warning}.get(f.severity, st.info)
            level(f"[{f.column or 'Dataset'}] {f.message}")
    else:
        st.success("No critical health flags detected.")
        
    st.divider()
    
    # Load sample df once for the whole tab
    sample_df = None
    if dco.reservoir_sample_path and os.path.exists(dco.reservoir_sample_path):
        sample_df = pd.read_parquet(dco.reservoir_sample_path)
        
    # --- Row 1: Full Width Data Sample ---
    st.markdown("#### Data Sample")
    if sample_df is not None:
        st.dataframe(sample_df.head(10), use_container_width=True)
    else:
        st.caption("No sample data available.")
            
    st.divider()
    
    # --- Row 2: Full Width Dictionary & Statistics ---
    st.markdown("#### Column Dictionary & Statistics")
    schema_data = []
    for col_name, prof in dco.columns.items():
        sample_series = sample_df[col_name] if sample_df is not None and col_name in sample_df.columns else None
        insight = _get_optimization_insight(prof, sample_series)
        
        mean_val = f"{prof.mean:.2f}" if prof.mean is not None else "-"
        std_val = f"{prof.std:.2f}" if prof.std is not None else "-"
        
        min_val = f"{prof.min_val}" if prof.min_val is not None else "-"
        max_val = f"{prof.max_val}" if prof.max_val is not None else "-"
        
        schema_data.append({
            "Column": col_name, 
            "Type": prof.dtype, 
            "Nulls": f"{prof.null_pct:.1%}",
            "Min": min_val,
            "Max": max_val,
            "Mean": mean_val,
            "Std": std_val,
            "Optimization": insight,
        })
        
    st.dataframe(pd.DataFrame(schema_data), hide_index=True, use_container_width=True)
        
    st.divider()
    st.markdown("#### Domain Context")
    st.caption("Fetch external context (Web/LLM) to help the agent understand this dataset.")
    render_category_checklist(None, "context")

    st.divider()
    st.markdown("#### Domain Knowledge & Context Injection")
    st.caption("Upload a data dictionary, markdown documentation, or notes file to teach the agent about your specific business rules.")
    
    uploaded_doc = st.file_uploader("Upload Data Dictionary (.txt, .md)", type=["txt", "md"], key="domain_doc_uploader")
    
    if uploaded_doc is not None:
        if "domain_context_loaded" not in st.session_state or st.session_state.get("loaded_doc_name") != uploaded_doc.name:
            with st.spinner("Analyzing documentation and mapping to columns..."):
                doc_text = uploaded_doc.read().decode("utf-8")
                
                # Fetch available column names from current dco
                col_names = list(dco.columns.keys())
                
                # Run the RAG parser
                from src.tools.rag_tools import parse_domain_dictionary
                mappings = parse_domain_dictionary(doc_text, col_names)
                
                # Update dco columns with descriptions
                for col_name, desc in mappings.items():
                    if col_name in dco.columns:
                        dco.columns[col_name].description = desc
                        
                st.session_state.domain_context_loaded = True
                st.session_state.loaded_doc_name = uploaded_doc.name
                st.success(f"Successfully mapped context for {len(mappings)} columns!")


def render_target_tab():
    dco = st.session_state.dco
    
    st.subheader("Target Configuration")
    
    # 1. ALWAYS RENDER THE SELECTION UI FIRST
    col_names = list(dco.columns.keys())
    # Remember the previous selection if it exists
    default_idx = col_names.index(dco.target.column) if dco.target.column in col_names else 0
    
    c1, c2 = st.columns([3, 1])
    with c1:
        selected_target = st.selectbox("Select Target Variable:", options=col_names, index=default_idx)
    with c2:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True) # Aligns button with selectbox
        if st.button("Confirm Target", type="primary", use_container_width=True):
            dco.target.column = selected_target
            dco.target.confirmed_by_user = True
            st.rerun()

    # 2. THE GUARD CLAUSE (Blocks analysis until confirmed)
    if not dco.target.column or not dco.target.confirmed_by_user:
        st.info("Please select and confirm a Target Column above to unlock target analysis.")
        return

    # 3. DOWNSTREAM ANALYSIS (Safe to proceed)
    target_col = dco.target.column
    prof = dco.columns[target_col]
    
    st.success(f"Confirmed Target: **{target_col}**")
    
    # --- TASK DETECTION ROUTER ---
    is_classification = prof.dtype.upper() in ["VARCHAR", "TEXT", "BOOLEAN", "CATEGORY"] or (prof.distinct_count is not None and prof.distinct_count < 20)
    
    if is_classification:
        st.info(f"**Task Detected:** Classification ({prof.distinct_count} unique classes)")
        
        # Paste your original class imbalance warning code here
        # (e.g., calculating the minority class percentage and showing a warning)
        
    else:
        st.info("**Task Detected:** Regression (Continuous Variable)")
        st.caption("Target is continuous. Displaying distribution metrics instead of class balance.")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Min", f"{prof.min_val:.2f}" if prof.min_val is not None else "-")
        col2.metric("Max", f"{prof.max_val:.2f}" if prof.max_val is not None else "-")
        col3.metric("Mean", f"{prof.mean:.2f}" if prof.mean is not None else "-")
        col4.metric("Std Dev", f"{prof.std:.2f}" if prof.std is not None else "-")
        
    st.divider()

    # 4. RENDER THE ANALYSIS TOOLS
    st.markdown(f"#### Target Analysis: `{target_col}`")
    
    target_processes = [s for s in REGISTRY.list() if s.requires_target]
    
    if not target_processes:
        st.warning("No target processes registered yet.")
        return

    selected_target_procs = []
    cols = st.columns(2)
    for i, spec in enumerate(target_processes):
        display_name = spec.name.replace("_", " ").title()
        with cols[i % 2]:
            if st.checkbox(f"{display_name} ({spec.cost.value})", value=True, help=spec.description, key=f"tgt_{spec.name}"):
                selected_target_procs.append(spec.name)

    if st.button("Run Target Analysis", type="primary", key="btn_run_target"):
        placeholders = {name: st.empty() for name in selected_target_procs}
        for name, result in run_selected(selected_target_procs, dco=dco):
            with placeholders[name].container():
                render_result(name, result)


def render_category_checklist(title: str, category: str):
    dco = st.session_state.dco
    if not dco: return

    if title:
        st.subheader(title)
        
    eligible = [
        s for s in REGISTRY.list()
        if s.category == category
        and not s.requires_target  # Exclude target-specific tools from general tabs
        and s.cost != ProcessCost.NETWORK
        and not (s.cost == ProcessCost.LLM and not os.getenv("GROQ_API_KEY"))
    ]

    if not eligible:
        st.info("No processes available for this category.")
        return

    selected = []
    cols = st.columns(2)
    for i, spec in enumerate(eligible):
        with cols[i % 2]:
            display_name = spec.name.replace("_", " ").title() # Clean Snake Case
            # Set value=True to check by default
            if st.checkbox(f"{display_name} ({spec.cost.value})", value=True, help=spec.description, key=f"chk_{spec.name}_{category}"):
                selected.append(spec.name)

    btn_key = f"run_btn_{category}"
    if st.button("Run selected", disabled=not selected, type="primary", key=btn_key):
        kwargs = {"dco": dco}
        if any(REGISTRY.get(n).cost == ProcessCost.LLM for n in selected):
            fast_llm = st.session_state.get("_test_fast_llm_override") or get_llm("fast")
            kwargs["llm_fn"] = lambda p: fast_llm.invoke(p).content

        placeholders = {name: st.empty() for name in selected}
        for name, result in run_selected(selected, **kwargs):
            st.session_state.process_results[name] = result
            with placeholders[name].container():
                render_result(name, result)
    else:
        for name, result in st.session_state.process_results.items():
            if REGISTRY.get(name).category == category:
                render_result(name, result)


def render_automl_tab():
    dco = st.session_state.dco
    if not dco:
        return

    st.subheader("Automated Machine Learning Suite")
    
    # 1. Target Column Guard Clause
    if not dco.target.column or not dco.target.confirmed_by_user:
        st.info("Please go to the **Target Analysis** tab and confirm your target variable before training models.")
        return

    st.markdown(f"Current Target Objective: `{dco.target.column}`")
    st.caption("This suite trains multiple baseline algorithms using a stratified train-test split on your reservoir data sample to build a comparative performance leaderboard.")
    
    st.divider()

    # 2. Action Trigger Button
    if st.button("Run AutoML Suite", type="primary", use_container_width=True):
        with st.spinner("Partitioning data, training algorithms, and calculating cross-metrics..."):
            try:
                # Direct invocation of the process tool bypasses checklist bottlenecks
                results = run_automl_suite(dco)
                st.session_state["automl_cache"] = results
            except Exception as e:
                st.error(f"AutoML Suite Execution Failed: {str(e)}")
                
    # 3. Persistent Render Cache Check
    if "automl_cache" in st.session_state:
        results = st.session_state["automl_cache"]
        
        # Pass directly to your custom renderer in ui/render.py
        render_result("run_automl_suite", results)  