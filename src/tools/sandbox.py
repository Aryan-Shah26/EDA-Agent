"""
Safe ad-hoc code execution - the long-tail fallback for requests the
Process Registry doesn't cover. Two mandatory layers, defense-in-depth
(neither alone is bulletproof against a determined adversary; together
they're appropriate for an EDA agent's actual threat model - buggy or
careless LLM-generated code, not hostile red-teaming):
  1. AST allowlist (validate_code) - rejects disallowed imports, dunder
     attribute access, and dangerous builtins BEFORE anything runs.
  2. Subprocess isolation (run_sandboxed) - even code that passes the
     allowlist runs in a separate process with kernel-enforced CPU/memory
     limits and a wall-clock timeout, so a gap in the allowlist can hang
     or balloon memory without taking down the agent process.
Operates on the reservoir sample (`df`, pre-loaded) by default - full-data
operations should go through profiler.query_full_data(), not here.
"""
import ast
import json
import os
import psutil
import subprocess
import sys
import uuid

try :
    import resource
except ImportError:
    resource = None

from ..config import CONFIG

ALLOWED_IMPORTS = {"pandas", "numpy", "math", "statistics", "json", "datetime", "re", "itertools", "collections"}
BANNED_CALLS = {
    "eval", "exec", "compile", "__import__", "open", "input",
    "vars", "globals", "locals", "getattr", "setattr", "delattr",
}

# Static, non-user-controlled. User code is read from a separate file at
# runtime (SANDBOX_CODE_PATH) rather than string-embedded here, so there's
# no template-injection/escaping surface between this script and the code
# it's about to run.
RUNNER_SCRIPT = '''
import os, json, traceback
import pandas as pd
import numpy as np

sample_path = os.environ["SANDBOX_SAMPLE_PATH"]
output_path = os.environ["SANDBOX_OUTPUT_PATH"]
code_path = os.environ["SANDBOX_CODE_PATH"]

with open(code_path) as f:
    user_code = f.read()

df = pd.read_parquet(sample_path)
safe_builtins = {
    "len": len, "range": range, "list": list, "dict": dict, "set": set,
    "tuple": tuple, "sum": sum, "min": min, "max": max, "sorted": sorted,
    "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
    "abs": abs, "round": round, "int": int, "float": float, "str": str,
    "bool": bool, "print": print, "isinstance": isinstance,
    "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError, "KeyError": KeyError,
}
safe_globals = {"__builtins__": safe_builtins, "pd": pd, "np": np, "df": df}

try:
    exec(compile(user_code, "<sandboxed_code>", "exec"), safe_globals)
    result = safe_globals.get("result")
    if hasattr(result, "to_dict"):
        result = result.to_dict()
    with open(output_path, "w") as f:
        json.dump({"success": True, "result": result}, f, default=str)
except Exception as e:
    with open(output_path, "w") as f:
        json.dump({"success": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}, f)
'''


def validate_code(code: str) -> list[str]:
    """
    Pure static analysis - never executes anything. Returns a list of
    violation messages; empty list means the code passed. Catches both the
    direct route (banned imports/calls) and the classic restricted-exec
    escape route (dynamic attribute access via getattr/setattr to reach
    __class__.__bases__...__subclasses__() - blocked by banning getattr/
    setattr outright, not just literal `.attr` dunder syntax).
    """
    violations = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"syntax error: {e}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    violations.append(f"disallowed import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORTS:
                violations.append(f"disallowed import: {node.module}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in BANNED_CALLS:
                violations.append(f"disallowed call: {node.func.id}()")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                violations.append(f"disallowed dunder attribute access: .{node.attr}")

    return violations


def _limit_resources():
    """preexec_fn for the sandbox subprocess: kernel-enforced CPU time and
    address-space ceilings (POSIX only) - a runaway loop or allocation gets
    killed by the OS, not just slowed down by Python-level bookkeeping."""
    cfg = CONFIG.sandbox
    resource.setrlimit(resource.RLIMIT_CPU, (cfg.max_cpu_seconds, cfg.max_cpu_seconds))
    mem_bytes = cfg.max_memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))


def run_sandboxed(code: str, dco, scratch_dir: str = None) -> dict:
    """
    Validates `code`; only if it passes does a subprocess get spawned at
    all. The subprocess gets `df` pre-loaded from dco's reservoir sample
    and must assign its answer to a variable named `result`
    (JSON-serializable, or a pandas object with .to_dict()).
    Returns {"success", "result"|"error", "violations"}.
    """
    violations = validate_code(code)
    if violations:
        return {"success": False, "violations": violations, "error": None, "result": None}

    if not dco.reservoir_sample_path:
        return {"success": False, "violations": [], "result": None,
                "error": "no reservoir sample available on this DataContextObject"}

    scratch_dir = scratch_dir or CONFIG.ingestion.sample_dir
    os.makedirs(scratch_dir, exist_ok=True)
    run_id = uuid.uuid4().hex
    code_path = os.path.join(scratch_dir, f"{run_id}_code.py")
    runner_path = os.path.join(scratch_dir, f"{run_id}_runner.py")
    output_path = os.path.join(scratch_dir, f"{run_id}_output.json")

    with open(code_path, "w") as f:
        f.write(code)
    with open(runner_path, "w") as f:
        f.write(RUNNER_SCRIPT)

    env = {
        **os.environ,
        "SANDBOX_SAMPLE_PATH": dco.reservoir_sample_path,
        "SANDBOX_OUTPUT_PATH": output_path,
        "SANDBOX_CODE_PATH": code_path,
    }

    try:
        proc = subprocess.run(
            [sys.executable, runner_path],
            env=env,
            timeout=CONFIG.sandbox.timeout_seconds,
            preexec_fn=_limit_resources if os.name == "posix" else None,
            capture_output=True, text=True,
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "violations": [], "result": None,
                "error": f"execution exceeded {CONFIG.sandbox.timeout_seconds}s timeout"}
    finally:
        for p in (code_path, runner_path):
            try:
                os.remove(p)
            except OSError:
                pass

    if not os.path.exists(output_path):
        # subprocess was killed before writing output - most likely RLIMIT_AS (OOM) or RLIMIT_CPU
        stderr_tail = (proc.stderr or "").strip()[-2000:]
        return {"success": False, "violations": [], "result": None,
                "error": stderr_tail or "sandboxed process produced no output (likely killed by a resource limit)"}

    with open(output_path) as f:
        payload = json.load(f)
    os.remove(output_path)
    payload.setdefault("violations", [])
    return payload


def run_with_self_correction(code: str, dco, fix_fn, max_attempts: int = None) -> dict:
    """
    Runs `code` sandboxed; on failure, calls fix_fn(code, feedback) ->
    corrected code and retries, up to max_attempts (CONFIG.sandbox default)
    before surfacing the last failure as-is. fix_fn is injected, not
    imported - same provider-agnostic pattern as context_lookup.py - so
    this module has no dependency on which LLM does the fixing.
    """
    max_attempts = CONFIG.sandbox.max_self_correct_attempts if max_attempts is None else max_attempts
    current_code = code
    result = None

    for attempt in range(max_attempts + 1):
        result = run_sandboxed(current_code, dco)
        if result["success"] or attempt == max_attempts:
            result["attempts"] = attempt + 1
            return result
        feedback = result.get("error") or "; ".join(result.get("violations", []))
        current_code = fix_fn(current_code, feedback)
