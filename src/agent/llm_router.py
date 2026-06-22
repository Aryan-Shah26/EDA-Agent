"""
Groq model router. API key is loaded from the environment (GROQ_API_KEY in
.env for local dev, a secrets manager / platform env var for deployment).
Never sourced from user input. Model IDs live in config.py so they're a
one-line edit when Groq's lineup changes.

[Unverified] model IDs - check console.groq.com/docs/models before running.
"""
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from ..config import CONFIG

load_dotenv()  # no-op if already loaded; safe to call multiple times


def get_api_key() -> str:
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "GROQ_API_KEY not set. Add it to your .env file (local) "
            "or as an environment variable / secret (deployment)."
        )
    return key


def get_llm(task: str = "reasoning", **kwargs) -> ChatGroq:
    """Returns a ChatGroq instance for the given task ('fast' | 'reasoning' | 'vision').
    API key is read from the environment, never passed in by callers."""
    models = CONFIG.llm.models
    if task not in models:
        raise ValueError(f"Unknown task '{task}', expected one of {list(models)}")
    return ChatGroq(model=models[task], api_key=get_api_key(), **kwargs)
