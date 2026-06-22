"""
Multimodal vision support. An uploaded image (chart, schema diagram, ER
diagram) is described ONCE by a vision model, and only the resulting TEXT
goes into DataContextObject.external_context - consistent with the rest of
this codebase's core rule that the chat agent never gets raw bytes/rows on
every turn, only the bounded, already-distilled context object.

vision_fn is injected (same decoupling pattern as context_lookup.py /
sandbox.py's fix_fn): this module has no import of `groq` or any specific
SDK. build_groq_vision_fn() below is one concrete adapter, not a
dependency - swap in a different one without touching describe_image().
"""
import base64
from typing import Callable

from ..ingestion.data_context import DataContextObject

DEFAULT_PROMPT = (
    "Describe this image in the context of a dataset's exploratory analysis. "
    "If it's a chart, describe what's plotted (axes, trend, notable outliers). "
    "If it's a schema/ER diagram, describe the tables and relationships. "
    "Be concise - 3-5 sentences."
)


def encode_image_base64(image_path: str) -> str:
    """Reads an image file and returns its base64-encoded content as a str."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def describe_image(
    image_path: str,
    vision_fn: Callable[[str, str, str], str],
    mime_type: str = "image/png",
    prompt: str = DEFAULT_PROMPT,
) -> str:
    """
    Encodes the image and calls vision_fn(base64_data, mime_type, prompt) ->
    description text. The image bytes never leave this function call - only
    the returned string is meant to persist anywhere.
    """
    b64 = encode_image_base64(image_path)
    return vision_fn(b64, mime_type, prompt)


def attach_image_context(dco: DataContextObject, description: str, label: str) -> None:
    """
    Appends one image's description to dco.external_context under
    "images" (a list, so multiple uploads accumulate rather than
    overwrite each other). Only ever stores text - this is the boundary
    where image content gets converted to a small, reusable string instead
    of being re-sent to the LLM on every subsequent turn.
    """
    if dco.external_context is None:
        dco.external_context = {}
    dco.external_context.setdefault("images", [])
    dco.external_context["images"].append({"label": label, "description": description})


def build_groq_vision_fn(llm) -> Callable[[str, str, str], str]:
    """
    Adapts a LangChain-compatible chat model (e.g. llm_router.get_llm("vision"))
    into the vision_fn signature describe_image() expects. Constructs the
    standard multimodal message format (text + image_url data URI) and
    returns just the text of the response.
    """
    from langchain_core.messages import HumanMessage

    def _vision_fn(b64_data: str, mime_type: str, prompt: str) -> str:
        message = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_data}"}},
        ])
        response = llm.invoke([message])
        return response.content

    return _vision_fn
