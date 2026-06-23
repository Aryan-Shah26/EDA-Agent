"""
DataContextObject: the only thing the LLM ever sees. Never raw rows, never
a full profile dump - a versioned, token-budgeted summary. All prompts are
built from .to_prompt_context(), never from a raw DataFrame.
"""
from dataclasses import dataclass, field
from typing import Any, Optional
import json
import time

from ..config import CONFIG


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    null_count: int = 0
    null_pct: float = 0.0
    distinct_count: Optional[int] = None  # approx for high-cardinality cols
    distinct_is_approx: bool = True
    min_val: Optional[Any] = None
    max_val: Optional[Any] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    skew: Optional[float] = None
    description: str = "No domain context provided."


@dataclass
class TargetStatus:
    column: Optional[str] = None
    confirmed_by_user: bool = False
    candidates: list = field(default_factory=list)  # [{column, score, reasons}]
    health: Optional[dict] = None  # filled by audit_target_health()


@dataclass
class Flag:
    code: str
    severity: str  # "info" | "warning" | "critical"
    message: str
    column: Optional[str] = None


@dataclass
class DataContextObject:
    source_name: str
    n_rows: Optional[int]
    n_cols: int
    columns: dict  # name -> ColumnProfile
    reservoir_sample_path: Optional[str] = None  # parquet path; never inlined into prompts
    target: TargetStatus = field(default_factory=TargetStatus)
    flags: list = field(default_factory=list)
    process_log: list = field(default_factory=list)  # names of completed registry processes
    external_context: Optional[dict] = None  # user-provided or looked-up dataset context
    created_at: float = field(default_factory=time.time)

    def add_flag(self, code: str, severity: str, message: str, column: Optional[str] = None):
        self.flags.append(Flag(code=code, severity=severity, message=message, column=column))

    def to_prompt_context(self, token_budget: int = None) -> str:
        """
        Priority order (highest first, dropped/truncated last):
          1. target status + health
          2. flags (critical > warning > info)
          3. schema
          4. process log (what's already been computed - avoids redundant tool calls)
          5. external context (user-provided / looked-up dataset background)
        ~4 chars/token estimate. Schema is truncated (not dropped) if it's the
        section that blows the budget, since it's needed for almost every task.
        token_budget defaults to CONFIG.data_context.token_budget if not passed.
        """
        token_budget = CONFIG.data_context.token_budget if token_budget is None else token_budget

        def est_tokens(s: str) -> int:
            return max(1, len(s) // 4)

        sections: list[tuple[str, str]] = []

        if self.target.column:
            t = self.target
            lines = [f"TARGET: {t.column} (confirmed_by_user={t.confirmed_by_user})"]
            if t.health:
                for f in t.health.get("flags", []):
                    lines.append(f"  - [{f['severity']}] {f['code']}: {f['detail']}")
                if t.health.get("recommendations"):
                    lines.append("  recommendations: " + "; ".join(t.health["recommendations"]))
            sections.append(("target", "\n".join(lines)))
        elif self.target.candidates:
            lines = ["TARGET: not yet confirmed by user. Top candidates:"]
            for c in self.target.candidates[:5]:
                lines.append(f"  - {c['column']} (score={c['score']}): {'; '.join(c['reasons'])}")
            sections.append(("target_candidates", "\n".join(lines)))

        if self.flags:
            order = {"critical": 0, "warning": 1, "info": 2}
            sorted_flags = sorted(self.flags, key=lambda f: order.get(f.severity, 3))
            lines = ["DATA FLAGS:"] + [
                f"  - [{f.severity}] {f.code} ({f.column or 'dataset'}): {f.message}"
                for f in sorted_flags
            ]
            sections.append(("flags", "\n".join(lines)))

        schema_lines = [f"SCHEMA ({self.n_rows} rows x {self.n_cols} cols):"]
        for name, p in self.columns.items():
            dn = f"{p.distinct_count}{'~' if p.distinct_is_approx else ''}" if p.distinct_count is not None else "?"
            schema_lines.append(f"  - {name}: {p.dtype}, null={p.null_pct:.1%}, distinct={dn}")
        sections.append(("schema", "\n".join(schema_lines)))

        if self.process_log:
            lines = ["ALREADY COMPUTED (do not redo without reason):"]
            lines += [f"  - {p}" for p in self.process_log[-10:]]
            sections.append(("process_log", "\n".join(lines)))

        if self.external_context:
            sections.append(
                ("external_context", "EXTERNAL CONTEXT:\n  " + json.dumps(self.external_context)[:800])
            )

        output_parts, running = [], 0
        for name, text in sections:
            cost = est_tokens(text)
            if running + cost > token_budget:
                if name == "schema":
                    remaining_chars = max(token_budget - running, 0) * 4
                    truncated = text[:remaining_chars]
                    output_parts.append(truncated + "\n  ... (schema truncated - more columns omitted)")
                    running += est_tokens(truncated)
                continue
            output_parts.append(text)
            running += cost

        return "\n\n".join(output_parts)
