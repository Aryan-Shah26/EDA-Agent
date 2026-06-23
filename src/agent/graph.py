"""
LangGraph chat orchestrator. The agent node builds its system prompt from
DataContextObject.to_prompt_context() (token-bounded, never raw data) plus
the running message history, and has whatever tools
tool_adapter.build_tools_for_session() exposed for this session. State is
checkpointed per thread_id via the SqliteSaver passed into build_graph, so
a session can be resumed.

HITL: when enable_hitl=True, interrupt_before=['human_approval'] pauses the
graph between the agent proposing a tool call and the ToolNode executing it.
The UI detects the pause (graph.get_state(config).next == ('human_approval',)),
renders the pending tool call for the user to approve/reject, then resumes
with Command(resume=True) (approved) or Command(resume=False) (rejected).
On rejection, a synthetic ToolMessage is added so the agent receives a
clean 'rejected' response and doesn't error on a dangling tool_call id.
"""
import os
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import StructuredTool

from ..ingestion.data_context import DataContextObject

os.environ.setdefault(
    "LANGGRAPH_MSGPACK_ALLOWED_MODULES",
    "src.ingestion.data_context",
)

SYSTEM_PROMPT_TEMPLATE = """You are a data analysis assistant for the dataset described below. \
Call a tool only when the question actually requires it, and prefer free tools over ones that \
call an LLM or external search. Never ask the user to paste raw data - you already have a \
profile of it below.

{data_context}

You write Python code to answer user questions about a pandas DataFrame named `df`.

--- CORE BEHAVIORS ---

1. VISUALIZATION RULES (No Unprompted Plots):
Default to answering with raw numbers, statistics, or DataFrames. 
ONLY generate a plot if the user explicitly asks to "plot", "graph", "visualize", or "draw", OR if they ask about "trends over time". 

2. SAFE STRING FILTERING (The Case-Insensitive Exact Match):
Users often miscapitalize categorical values. Do NOT use standard exact matching (`df['col'] == 'Apple'`) because it fails on 'apple'. 
Do NOT use `.str.contains()` for exact lookups, as it causes substring contamination (e.g., 'US' matches 'Russia').
Instead, normalize both sides to lowercase for safe exact matching:
`df[df['col'].astype(str).str.lower() == 'target_value'.lower()]`
*Only* use `.str.contains()` if the user explicitly asks for a partial match (e.g., "contains", "includes", "starts with").

3. THE "EMPTY RESULT" PROTOCOL:
If you filter the DataFrame and the resulting DataFrame is empty, DO NOT hallucinate an answer. You must output the code you ran, and then politely state: "No data was found matching that exact criteria. Would you like me to check for similar names?"

4. COMPLEXITY BREAKDOWN:
If a query requires multiple steps (filtering, grouping, aggregating), break the code into intermediate variables. Do not write massive one-liners.

5. MANDATORY CODE TRANSPARENCY:
You must ALWAYS show the Python code you are writing inside a markdown block:
```python
# Your code here

6. PANDAS 2.0+ VALUE_COUNTS COMPATIBILITY:
When plotting value counts, NEVER assume the columns will be named 'index' after running `.reset_index()`. Modern Pandas outputs columns named `[column_name, 'count']`. 
To ensure your code never crashes across versions, ALWAYS explicitly rename the columns immediately after resetting the index:
```python
counts_df = df['column_name'].value_counts().reset_index()
counts_df.columns = ['value', 'count']
fig = px.bar(counts_df, x='value', y='count')

1. STRICT OBJECTIVE ALIGNMENT (Scalar vs. Visual):
You must match your code's final output to the semantic type of the user's question.
- SCALAR QUERIES: If the user asks "what is the average...", "how many...", "what percentage...", or any question requiring a single numeric or text answer, you MUST ONLY calculate that number and assign it to a variable named `result`. Do NOT create a `fig`, `ax`, or any plot.
- VISUAL QUERIES: Only create a chart or plot if the user explicitly uses words like "plot", "chart", "graph", "visualize", or "draw".

If you create a plot for a scalar query, the sandbox wrapper will reject your code and count it as a failure.

2. CHANNALING FINAL ANSWERS:
Your code must store its final answer in a variable named `result`. 
- If the question expects a number, `result` must be that number/string (e.g., `result = f"{df['Survived'].mean() * 100:.2f}%"`).
- If the question expects a plot, `result` must be assigned the figure object (e.g., `result = px.bar(...)`).
Do not generate both. Choose ONE based strictly on the user's intent. 
"""


class AgentState(MessagesState):
    dco: DataContextObject
    __approved: bool = False  # written by human_approval_node, read by approval_router


def build_graph(
    llm: BaseChatModel,
    tools: list[StructuredTool],
    checkpointer=None,
    enable_hitl: bool = False,
):
    """
    Wires the agent<->tools loop. When enable_hitl=True, a human_approval node is inserted between agent and tools; the graph pauses there via interrupt(), surfacing the pending tool call(s) to whatever calls graph.get_state(config). The caller resumes with:
      - Command(resume=True)  -> tool executes, agent continues
      - Command(resume=False) -> synthetic ToolMessage injected, agent sees 'rejected' and replies gracefully

    A checkpointer is REQUIRED for HITL (graph state must survive betweenthe pause and the resume call).
    """
    if enable_hitl and checkpointer is None:
        raise ValueError("enable_hitl=True requires a checkpointer - graph state must persist between interrupt and resume")

    llm_with_tools = llm.bind_tools(tools) if tools else llm

    def agent_node(state: AgentState):
        """Builds the system prompt fresh from the current DataContextObject
        (captures any updates from prior tool calls), then invokes the LLM."""
        dco: DataContextObject = state["dco"]
        system = SystemMessage(content=SYSTEM_PROMPT_TEMPLATE.format(
            data_context=dco.to_prompt_context()))
        response = llm_with_tools.invoke([system] + state["messages"])
        return {"messages": [response]}

    def human_approval_node(state: AgentState):
        """
        Pauses via interrupt(), surfacing pending tool_calls to the UI.
        The interrupt payload is {pending_tool_calls: [...]} - the UI can render each tool name + args for the user to inspect.
        On resume:
          True  -> sets __approved=True, approval_router sends to tools
          False -> injects ToolMessage(rejected) per pending call so the agent gets a clean response without a dangling    tool_call id, then routes back to agent
        """
        last_msg = state["messages"][-1]
        approved = interrupt({"pending_tool_calls": last_msg.tool_calls})
        if not approved:
            rejection_messages = [
                ToolMessage(
                    content=(
                        "USER OVERRIDE: This tool call was explicitly REJECTED by the user. "
                        "DO NOT retry this tool. DO NOT propose it again. "
                        "Acknowledge the rejection and provide a fallback textual response instead."
                    ),
                    tool_call_id=tc["id"],
                    name=tc["name"],
                )
                for tc in last_msg.tool_calls
            ]
            return {"messages": rejection_messages, "__approved": False}
            
        return {"__approved": True}

    def approval_router(state: AgentState) -> str:
        return "tools" if state.get("__approved", False) else "agent"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_edge(START, "agent")

    if tools:
        graph.add_node("tools", ToolNode(tools))
        if enable_hitl:
            graph.add_node("human_approval", human_approval_node)
            graph.add_conditional_edges(
                "agent", tools_condition,
                {"tools": "human_approval", "__end__": END}
            )
            graph.add_conditional_edges(
                "human_approval", approval_router,
                {"tools": "tools", "agent": "agent"}
            )
            graph.add_edge("tools", "agent")
            return graph.compile(
                checkpointer=checkpointer,
                interrupt_before=["human_approval"],
            )
        else:
            graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", "__end__": END})
            graph.add_edge("tools", "agent")
    else:
        graph.add_edge("agent", END)

    return graph.compile(checkpointer=checkpointer)
