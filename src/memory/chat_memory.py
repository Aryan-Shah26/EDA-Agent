"""
Rolling chat memory: keeps the last `keep_recent` turns verbatim and folds
older ones into a running summary via an injected summarize_fn, so a long
session's prompt doesn't grow unbounded. This is independent of the
SqliteSaver checkpointer in checkpointer.py - the checkpointer persists the
FULL history to disk (cheap, useful for resuming/auditing); ChatMemory
controls what actually gets sent to the LLM each turn (expensive, must stay
bounded). summarize_fn is injected, not imported, for the same provider-
agnostic reason as context_lookup.py.
"""
from dataclasses import dataclass, field
from ..config import CONFIG


@dataclass
class ChatMemory:
    keep_recent: int = None  # resolved to CONFIG.memory.keep_recent in __post_init__ if unset
    summary: str = ""
    recent: list = field(default_factory=list)  # [{"role": ..., "content": ...}]

    def __post_init__(self):
        if self.keep_recent is None:
            self.keep_recent = CONFIG.memory.keep_recent

    def add(self, role: str, content: str, summarize_fn=None):
        """
        Append one turn. If this pushes `recent` past keep_recent, the
        oldest overflow turns are folded into `summary` via summarize_fn
        (one LLM call covering all overflow at once, not per-message) and
        then dropped from `recent`. If summarize_fn isn't injected, overflow
        is simply dropped - degraded but not silently wrong, since the full
        history still lives in the SqliteSaver checkpoint regardless.
        """
        self.recent.append({"role": role, "content": content})
        if len(self.recent) > self.keep_recent:
            cutoff = len(self.recent) - self.keep_recent
            overflow, self.recent = self.recent[:cutoff], self.recent[cutoff:]
            if summarize_fn is not None:
                overflow_text = "\n".join(f"{m['role']}: {m['content']}" for m in overflow)
                prompt = (
                    f"Existing summary: {self.summary}\n\nNew messages to fold in:\n{overflow_text}\n\n"
                    "Produce an updated concise summary covering all of the above."
                )
                self.summary = summarize_fn(prompt)
            # No summarize_fn injected -> overflow is dropped (degraded mode, not silent corruption:
            # the full history still exists in the SqliteSaver checkpoint).

    def as_context(self) -> str:
        return f"CONVERSATION SUMMARY SO FAR:\n{self.summary}" if self.summary else ""
