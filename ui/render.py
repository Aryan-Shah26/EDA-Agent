"""
Renders one process result dict based on its declared "type" field, the
same type strings tools/builtin_processes.py, feature_model/*.py, and
tools/context_lookup.py already emit - keeps rendering fully decoupled
from computation (a result can be rendered here, fed to the chat agent as
text, or asserted on in a test, all from the same dict).
"""
import streamlit as st
import pandas as pd


def render_result(name: str, result):
    st.markdown(f"**{name}**")

    if isinstance(result, Exception):
        st.error(f"{type(result).__name__}: {result}")
        return

    if not isinstance(result, dict):
        st.write(result)
        return

    if result.get("note"):
        st.caption(result["note"])

    rtype = result.get("type")
    if rtype == "table":
        _render_table(result)
    elif rtype == "heatmap":
        _render_heatmap(result)
    elif rtype == "multi_chart":
        _render_multi_chart(result)
    elif rtype == "text":
        st.write(result.get("data", ""))
    else:
        st.json(result)


def _render_table(result):
    data = result.get("data", [])
    if data:
        st.dataframe(pd.DataFrame(data), width='stretch')
    elif "outlier_pct" in result:
        st.metric("Outlier rate", f"{result.get('outlier_pct', 0):.2%}",
                   help=f"{result.get('outlier_count', 0)} rows flagged")
        contributors = result.get("top_contributing_columns", [])
        if contributors:
            st.dataframe(pd.DataFrame(contributors), width='stretch')
    elif not result.get("note"):
        st.caption("No data.")


def _render_heatmap(result):
    columns, matrix = result.get("columns", []), result.get("matrix", [])
    if not columns:
        return
    df = pd.DataFrame(matrix, columns=columns, index=columns)
    st.dataframe(df.style.background_gradient(cmap="coolwarm", vmin=-1, vmax=1).format("{:.2f}"),
                 width='stretch')


def _render_multi_chart(result):
    for col, chart in result.get("data", {}).items():
        st.caption(col)
        if chart["kind"] == "histogram":
            edges = chart["bin_edges"]
            midpoints = [f"{(edges[i] + edges[i + 1]) / 2:.1f}" for i in range(len(edges) - 1)]
            st.bar_chart(pd.DataFrame({"count": chart["counts"]}, index=midpoints))
        elif chart["kind"] == "bar":
            st.bar_chart(pd.DataFrame({"count": chart["values"]}, index=chart["labels"]))
