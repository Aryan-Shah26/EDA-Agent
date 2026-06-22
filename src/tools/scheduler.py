"""
Dashboard scheduler. Runs a user-selected checklist of registered processes
concurrently and yields (name, result) pairs in COMPLETION order, not
submission order - this is what lets the UI render each visualization the
moment it's ready instead of blocking on the slowest one.

ThreadPoolExecutor, not ProcessPoolExecutor: processes operate on the
reservoir sample (<=20k rows held by each worker independently from
parquet), pandas/numpy/sklearn release the GIL for most of their work, and
threads avoid the pickling overhead of shipping DataContextObject across
process boundaries. Revisit if a registered process turns out to be
genuinely CPU-bound pure Python.

A failure in one process is isolated and yielded as an exception - it must
never take down the other processes in the same batch.
"""
import concurrent.futures as cf
from .registry import REGISTRY


def run_selected(process_names: list[str], **kwargs):
    """
    Run each named process from REGISTRY concurrently with the same kwargs
    (typically dco=, llm_fn=, search_fn=). Yields (name, result) as each
    finishes - completion order, not submission order - so a caller (e.g.
    the Streamlit dashboard) can render each result the instant it's ready
    instead of waiting for the slowest process in the batch. A process that
    raises yields (name, exception) instead of crashing the whole batch.
    """
    if not process_names:
        return
    with cf.ThreadPoolExecutor(max_workers=min(8, len(process_names))) as ex:
        future_to_name = {
            ex.submit(REGISTRY.get(name).fn, **kwargs): name for name in process_names
        }
        for future in cf.as_completed(future_to_name):
            name = future_to_name[future]
            try:
                yield name, future.result()
            except Exception as e:  # noqa: BLE001 - intentionally broad, isolates per-process failure
                yield name, e
