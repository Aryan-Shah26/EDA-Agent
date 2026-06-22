"""
Process Registry: every visualization, statistic, model suggestion, and
external lookup is registered here as an independent unit with a declared
cost. Two consumers pull from this same registry and never block each other:
  - Dashboard scheduler (scheduler.py): user checks a list, selected
    processes run in parallel, each result renders as soon as it's ready.
  - Chat agent: calls a process by name on demand, lazily, only when a
    question actually needs it.
Nothing with LLM or network cost runs unless one of these two paths
explicitly asks for it - this is what keeps token/API usage proportional
to what the user actually wants, not to what the agent guesses they want.
"""
from dataclasses import dataclass
from typing import Callable, Any, Optional
from enum import Enum


class ProcessCost(Enum):
    FREE = "free"        # pandas/duckdb/sklearn on the sample - no LLM, no network
    LLM = "llm"           # calls an LLM
    NETWORK = "network"   # calls an external API (web search, etc.)


@dataclass
class ProcessSpec:
    name: str
    description: str
    fn: Callable[..., Any]
    cost: ProcessCost
    requires_target: bool = False
    category: str = "general"  # "visualization" | "statistic" | "model_suggestion" | "context"


class ProcessRegistry:
    def __init__(self):
        self._processes: dict[str, ProcessSpec] = {}

    def register(self, spec: ProcessSpec):
        if spec.name in self._processes:
            raise ValueError(f"Process '{spec.name}' already registered")
        self._processes[spec.name] = spec

    def get(self, name: str) -> ProcessSpec:
        return self._processes[name]

    def list(self, category: Optional[str] = None) -> list[ProcessSpec]:
        specs = list(self._processes.values())
        return [s for s in specs if s.category == category] if category else specs


REGISTRY = ProcessRegistry()


def process(name: str, description: str, cost: ProcessCost, requires_target: bool = False, category: str = "general"):
    """Decorator: registers fn under `name` at import time."""
    def wrapper(fn):
        REGISTRY.register(ProcessSpec(name, description, fn, cost, requires_target, category))
        return fn
    return wrapper
