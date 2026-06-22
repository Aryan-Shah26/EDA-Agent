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
import os
import tempfile
import streamlit as st

from src.ingestion.profiler import profile_dataset, get_class_counts
from src.target_analysis.detector import detect_target_candidates
from src.target_analysis.health_audit import audit_target_health
from src.tools.registry import REGISTRY, ProcessCost
from src.tools.scheduler import run_selected
from src.agent.llm_router import get_llm
from .render import render_result


def render_upload_section():
    uploaded = st.file_uploader("Upload a CSV or Parquet file", type=["csv", "parquet"])
    if uploaded is None:
        return

    suffix = os.path.splitext(uploaded.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        csv_path = tmp.name

    if st.session_state.csv_path != csv_path:
        with st.spinner("Profiling dataset (DuckDB, out-of-core - safe for large files)..."):
            st.session_state.dco = profile_dataset(csv_path)
        st.session_state.csv_path = csv_path
        st.session_state.process_results = {}


def render_target_section():
    dco = st.session_state.dco
    if dco is None:
        return

    st.subheader("Target column")

    if dco.target.confirmed_by_user and dco.target.column:
        st.success(f"Target: **{dco.target.column}**")
        for f in (dco.target.health or {}).get("flags", []):
            level = {"critical": st.error, "warning": st.warning}.get(f["severity"], st.info)
            level(f["detail"])
        if st.button("Change target"):
            dco.target.confirmed_by_user = False
            dco.target.column = None
            st.rerun()
        return

    candidates = dco.target.candidates or detect_target_candidates(dco)
    dco.target.candidates = candidates
    candidate_names = [c["column"] for c in candidates]
    other_columns = [c for c in dco.columns if c not in candidate_names]
    options = ["-- none --"] + candidate_names + other_columns

    choice = st.selectbox(
        "Does this dataset have a target/label column to predict?",
        options, index=1 if candidates else 0,
    )
    if choice != "-- none --" and st.button("Confirm target"):
        class_counts = get_class_counts(st.session_state.csv_path, choice)
        health = audit_target_health(dco, choice, class_counts=class_counts)
        dco.target.column = choice
        dco.target.confirmed_by_user = True
        dco.target.health = health
        st.rerun()


def render_process_checklist():
    dco = st.session_state.dco
    if dco is None:
        return

    st.subheader("Run analyses")
    eligible = [
        s for s in REGISTRY.list()
        if not (s.requires_target and not dco.target.column)
        and s.cost != ProcessCost.NETWORK
        and not (s.cost == ProcessCost.LLM and not os.getenv("GROQ_API_KEY"))
    ]

    selected = []
    cols = st.columns(2)
    for i, spec in enumerate(eligible):
        with cols[i % 2]:
            if st.checkbox(f"{spec.name} ({spec.cost.value})", help=spec.description, key=f"chk_{spec.name}"):
                selected.append(spec.name)

    if st.button("Run selected", disabled=not selected):
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
            render_result(name, result)
