"""
Import this once from any entry point (UI, agent setup, scripts) to
register every built-in process with the Process Registry. @process
decorators only take effect when their module is actually imported -
without a single place that imports all of them, the registry's contents
depend on import order/accident, which is exactly the kind of bug this
file exists to eliminate (caught via AppTest: the dashboard checklist
silently rendered zero checkboxes because nothing had imported the
process modules).
"""
from .tools import builtin_processes  # noqa: F401
from .tools import context_lookup     # noqa: F401
from .feature_model import feature_engineering  # noqa: F401
from .feature_model import model_selection      # noqa: F401
