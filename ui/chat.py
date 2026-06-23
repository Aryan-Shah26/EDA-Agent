import streamlit as st
from src.agent.code_copilot import run_with_self_correction
from src.agent.llm_router import get_llm

def render_chat_section():
    st.subheader("AI Co-Pilot")
    st.caption("Ask analytical questions or request visualizations. The agent will write, test, and self-correct Pandas code in the background.")
    
    # Ensure chat history exists
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    # 1. Render history container (Fixed height so it scrolls)
    chat_container = st.container(height=500)
    
    with chat_container:
        for turn in st.session_state.chat_messages:
            with st.chat_message(turn["role"]):
                if turn.get("type") == "chart":
                    st.plotly_chart(turn["data"], use_container_width=True)
                else:
                    st.write(turn["content"])
                    
                if turn.get("code"):
                    with st.expander("View Code", expanded=False):
                        st.code(turn["code"], language="python")

    # 2. Render Input
    user_input = st.chat_input("E.g., Plot a histogram of passenger ages...")
    
    if user_input:
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        
        with chat_container:
            with st.chat_message("user"):
                st.write(user_input)
                
            with st.chat_message("assistant"):
                # Setup the LLM and pass it to the copilot
                fast_llm = st.session_state.get("_test_fast_llm_override") or get_llm("fast")
                llm_fn = lambda p: fast_llm.invoke(p).content
                dco = st.session_state.dco
                
                # --- The Magic Self-Correcting UI ---
                with st.status("Writing and testing code...", expanded=True) as status:
                    final_result = None
                    
                    # Stream the generator we built in the backend
                    for step in run_with_self_correction(user_input, dco, llm_fn):
                        if step.get("status") == "retrying":
                            st.write(f"**Attempt {step['attempt']} failed:** `{step['error']}`. Rewriting code...")
                        elif step.get("status") in ["success", "failed"]:
                            final_result = step
                            break
                            
                    if final_result["status"] == "success":
                        status.update(label=f"Code executed successfully after {final_result['attempts']} attempt(s)!", state="complete", expanded=False)
                    else:
                        status.update(label="Failed to generate working code.", state="error", expanded=True)

                # --- Display Final Output ---
                if final_result["status"] == "success":
                    if final_result["type"] == "chart":
                        st.plotly_chart(final_result["data"], use_container_width=True)
                        msg_data = {"role": "assistant", "type": "chart", "data": final_result["data"], "code": final_result["code"]}
                    else:
                        st.write(final_result["data"])
                        msg_data = {"role": "assistant", "content": final_result["data"], "code": final_result["code"]}
                        
                    st.session_state.chat_messages.append(msg_data)
                else:
                    st.error(final_result["error"])
                    with st.expander("View Failing Code"):
                        st.code(final_result["code"])