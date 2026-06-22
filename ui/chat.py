"""
Chat section. Handles two modes based on enable_hitl:
  - HITL off: standard invoke -> response displayed immediately.
  - HITL on:  first invoke() pauses at human_approval; the pending tool
              call(s) are rendered in an approve/reject widget. On user
              action, Command(resume=True/False) continues the graph.

API key is loaded from environment via llm_router.get_llm() - not from
session state.
"""
import streamlit as st
from langgraph.types import Command
from langchain_core.messages import HumanMessage

from src.agent.llm_router import get_llm
from src.agent.tool_adapter import build_tools_for_session
from src.agent.graph import build_graph
from src.agent.checkpointer import get_checkpointer
from src.memory.chat_memory import ChatMemory


def _get_chat_llm():
    return st.session_state.get("_test_llm_override") or get_llm("reasoning")


def _get_graph():
    """Builds (or returns cached) the compiled graph for this session.
    Cached so we don't re-compile on every Streamlit rerun."""
    if not st.session_state.get("compiled_graph"):
        dco = st.session_state.dco
        llm = _get_chat_llm()
        tools = build_tools_for_session(
            dco, enable_sandbox=st.session_state.get("enable_sandbox", False))
        checkpointer = get_checkpointer()
        enable_hitl = st.session_state.get("enable_hitl", False)
        st.session_state.compiled_graph = build_graph(
            llm, tools, checkpointer=checkpointer, enable_hitl=enable_hitl)
    return st.session_state.compiled_graph


def _graph_config():
    return {"configurable": {"thread_id": st.session_state.session_id}}


def _is_paused_for_approval() -> bool:
    try:
        snap = _get_graph().get_state(_graph_config())
        return snap.next == ("human_approval",)
    except Exception:
        return False


def _render_pending_approval():
    """Renders pending tool call(s) with approve/reject buttons.
    Returns True if the user acted (so the caller can rerun), False if not."""
    snap = _get_graph().get_state(_graph_config())
    pending_calls = getattr(snap.values["messages"][-1], "tool_calls", [])

    st.warning("Agent wants to call a tool — approve or reject:")
    for tc in pending_calls:
        with st.expander(f"🔧 `{tc['name']}`", expanded=True):
            st.json(tc["args"]) if tc.get("args") else st.caption("No arguments.")

    col1, col2 = st.columns(2)
    approved = None
    with col1:
        if st.button("✅ Approve", key="hitl_approve"):
            approved = True
    with col2:
        if st.button("❌ Reject", key="hitl_reject"):
            approved = False

    if approved is None:
        return False

    with st.spinner("Running..." if approved else "Cancelling..."):
        result = _get_graph().invoke(Command(resume=approved), config=_graph_config())

    last = result["messages"][-1]
    
    # 1. If accepted and LLM responded normally
    if last.type == "ai" and last.content:
        st.session_state.chat_messages.append({"role": "assistant", "content": last.content})
        st.session_state.chat_memory.add("assistant", last.content)
        
    # 2. If rejected, the graph ends on a ToolMessage. Handle it gracefully.
    elif not approved:
        st.session_state.chat_messages.append({"role": "assistant", "content": "❌ Tool execution cancelled by user."})
        st.session_state.chat_memory.add("assistant", "❌ Tool execution cancelled by user.")

    return True


def render_chat_section():
    dco = st.session_state.dco
    if dco is None:
        st.info("Upload a dataset to start chatting about it.")
        return

    st.subheader("Chat")
    if st.session_state.chat_memory is None:
        st.session_state.chat_memory = ChatMemory()

    # Invalidate cached graph when HITL toggle changes
    if st.session_state.get("_last_hitl") != st.session_state.get("enable_hitl"):
        st.session_state.compiled_graph = None
        st.session_state["_last_hitl"] = st.session_state.get("enable_hitl")

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if _is_paused_for_approval():
        if _render_pending_approval():
            st.rerun()
        return

    user_input = st.chat_input("Ask about this dataset...")
    if not user_input:
        return

    st.session_state.chat_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.chat_memory.add("user", user_input)

    graph = _get_graph()
    config = _graph_config()

    with st.spinner("Thinking..."):
        result = graph.invoke(
            {"messages": [HumanMessage(content=user_input)], "dco": dco},
            config=config,
        )

    snap = graph.get_state(config)
    if snap.next == ("human_approval",):
        st.rerun()
        return

    last_msg = result["messages"][-1]
    response_text = last_msg.content

    # Fallback: if the agent's closing message is empty (it decided the tool
    # result speaks for itself), surface the last ToolMessage content instead.
    if not response_text:
        tool_msgs = [m for m in result["messages"] if hasattr(m, "name") and m.name == "run_python_code"]
        if tool_msgs:
            import json as _json
            try:
                payload = _json.loads(tool_msgs[-1].content)
                response_text = f"```\n{_json.dumps(payload.get('result', payload), indent=2, default=str)}\n```"
            except Exception:
                response_text = tool_msgs[-1].content

    if response_text:
        st.session_state.chat_messages.append({"role": "assistant", "content": response_text})
        with st.chat_message("assistant"):
            st.write(response_text)
        st.session_state.chat_memory.add("assistant", response_text)