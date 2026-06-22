"""
Decoupled dataset-context tools. Only run if the user hasn't supplied their
own description AND explicitly opts in (both cost LLM/network calls). Two
independent strategies - either can be selected or skipped on its own:
  - web_context_lookup: search the web for the dataset's origin / known schema
  - llm_context_lookup: ask an LLM to infer domain context from schema alone

Both take their provider as an injected callable (search_fn / llm_fn) rather
than importing a specific SDK - keeps this module testable with a mock and
swappable (Groq today, anything else later) without touching this file.
Results land in dco.external_context, never silently merged elsewhere.
"""
from .registry import process, ProcessCost
from ..ingestion.data_context import DataContextObject


@process(
    name="web_context_lookup",
    description="Web-search the dataset's source/column names for domain context.",
    cost=ProcessCost.NETWORK,
    category="context",
)
def web_context_lookup(dco: DataContextObject, search_fn=None, **_):
    if search_fn is None:
        raise ValueError("web_context_lookup requires a search_fn to be injected")
    query = f"{dco.source_name} dataset columns: {', '.join(list(dco.columns.keys())[:10])}"
    return {"type": "text", "data": search_fn(query)}


@process(
    name="llm_context_lookup",
    description="Ask an LLM to infer the likely domain/meaning of the dataset from its schema alone.",
    cost=ProcessCost.LLM,
    category="context",
)
def llm_context_lookup(dco: DataContextObject, llm_fn=None, **_):
    if llm_fn is None:
        raise ValueError("llm_context_lookup requires an llm_fn to be injected")
    schema_desc = ", ".join(f"{n} ({p.dtype})" for n, p in dco.columns.items())
    prompt = (
        f"Dataset '{dco.source_name}' has columns: {schema_desc}. "
        "In 3-4 sentences, what domain is this likely from and what do the columns probably represent?"
    )
    return {"type": "text", "data": llm_fn(prompt)}
