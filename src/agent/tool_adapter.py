"""
Converts ProcessRegistry entries into LangChain StructuredTool objects bound
to one session's DataContextObject (and, for context-lookup processes, an
injected llm_fn/search_fn). LLM-cost or network-cost processes are silently
excluded from the tool list if their provider wasn't injected, so the agent
never sees a tool that's guaranteed to raise on call - decoupling at the
boundary, not a runtime error the LLM has to recover from.
"""
from typing import Optional, Callable
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..tools.registry import REGISTRY, ProcessSpec, ProcessCost
from ..tools.sandbox import run_sandboxed, run_with_self_correction
from ..ingestion.data_context import DataContextObject

_COST_NOTE = {
    ProcessCost.FREE: "no extra cost",
    ProcessCost.LLM: "calls an LLM - only use if the question actually needs it",
    ProcessCost.NETWORK: "calls an external search API - only use if needed",
}


class _NoArgs(BaseModel):
    pass


def _make_tool(spec: ProcessSpec, dco: DataContextObject, llm_fn, search_fn) -> StructuredTool:
    """Wraps one ProcessSpec as a zero-arg LangChain tool, with dco (and
    llm_fn/search_fn, if the process needs them) bound via closure rather
    than exposed as LLM-fillable parameters - the agent decides WHETHER to
    call the tool, never WHAT data it operates on."""
    def _run() -> dict:
        kwargs = {"dco": dco}
        if spec.cost == ProcessCost.LLM:
            kwargs["llm_fn"] = llm_fn
        if spec.cost == ProcessCost.NETWORK:
            kwargs["search_fn"] = search_fn
        return spec.fn(**kwargs)

    return StructuredTool.from_function(
        func=_run,
        name=spec.name,
        description=f"{spec.description} ({_COST_NOTE[spec.cost]})",
        args_schema=_NoArgs,
    )


class _CodeArgs(BaseModel):
    code: str = Field(description=(
        "Python code operating on a pre-loaded DataFrame `df`. Assign the answer to a "
        "variable named `result`. Only pandas, numpy, math, statistics, json, datetime, re, "
        "itertools, collections may be imported; no file/network/system access."
    ))


def build_sandbox_tool(dco: DataContextObject, fix_fn: Optional[Callable[[str, str], str]] = None) -> StructuredTool:
    """
    Exposes the AST-validated, subprocess-isolated executor (sandbox.py) as
    a tool for requests the registered processes don't cover. If fix_fn is
    injected, a failed run gets automatic self-correction attempts before
    the agent ever sees the failure (CONFIG.sandbox.max_self_correct_attempts).
    Kept separate from build_tools_for_session's default tool list - higher
    risk than a fixed registered function, so it's opt-in via enable_sandbox=
    rather than always offered.
    """
    def _run(code: str) -> dict:
        if fix_fn is not None:
            return run_with_self_correction(code, dco, fix_fn)
        return run_sandboxed(code, dco)

    return StructuredTool.from_function(
        func=_run,
        name="run_python_code",
        description=(
            "Run custom Python analysis code on the dataset sample ONLY when no other tool "
            "covers the request - prefer registered tools whenever one already does the job."
        ),
        args_schema=_CodeArgs,
    )


def build_tools_for_session(
    dco: DataContextObject,
    llm_fn: Optional[Callable[[str], str]] = None,
    search_fn: Optional[Callable[[str], str]] = None,
    categories: Optional[list[str]] = None,
    enable_sandbox: bool = False,
    sandbox_fix_fn: Optional[Callable[[str, str], str]] = None,
) -> list[StructuredTool]:
    """
    Builds the tool list for one chat session. LLM-cost processes are
    skipped if llm_fn wasn't injected, NETWORK-cost ones if search_fn
    wasn't - so the model only ever sees tools that will actually work,
    instead of finding out via a ValueError after it decides to call one.
    The sandbox code-execution tool is OFF by default (enable_sandbox=True
    to include it) since it's higher-risk than a fixed registered function.
    """
    tools = []
    for spec in REGISTRY.list():
        if categories and spec.category not in categories:
            continue
        if spec.cost == ProcessCost.LLM and llm_fn is None:
            continue
        if spec.cost == ProcessCost.NETWORK and search_fn is None:
            continue
        tools.append(_make_tool(spec, dco, llm_fn, search_fn))

    if enable_sandbox:
        tools.append(build_sandbox_tool(dco, fix_fn=sandbox_fix_fn))

    return tools
