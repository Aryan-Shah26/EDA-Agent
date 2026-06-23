"""
Renders one process result dict based on its declared "type" field.
"""
import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt

def render_result(name: str, result):
    # Convert snake_case to Title Case (e.g., 'missingness_report' -> 'Missingness Report')
    display_title = name.replace("_", " ").title()
    
    st.divider()
    st.subheader(f"🔹 {display_title}")

    if isinstance(result, Exception):
        st.error(f"{type(result).__name__}: {result}")
        return

    if not isinstance(result, dict):
        st.write(result)
        return

    if result.get("note"):
        st.caption(result["note"])

    if not result:
        return
        
    rtype = result.get("type")
    
    if rtype == "text":
        st.markdown(result.get("data"))
    elif rtype == "automl_results":
        _render_automl_results(result)
    elif rtype == "table":
        _render_table(result)
    elif rtype == "heatmap":
        _render_heatmap(result)
    elif rtype == "multi_chart":
        _render_multi_chart(result)
    elif rtype == "multi_chart_grouped":         
        _render_multi_chart_grouped(result)      
    elif rtype == "text":
        st.write(result.get("data", ""))
    elif rtype == "altair_boxplots":
        _render_altair_boxplots(result)
    elif rtype == "null_matrix":
        _render_null_matrix(result)

    else:
        st.json(result)


def _render_heatmap(result):
    columns, matrix = result.get("columns", []), result.get("matrix", [])
    if not columns:
        return
    df = pd.DataFrame(matrix, columns=columns, index=columns)
    st.dataframe(df.style.background_gradient(cmap="coolwarm", vmin=-1, vmax=1).format("{:.2f}"),
                 width='stretch')

def _render_table(result):
    data = result.get("data", [])
    if data:
        df = pd.DataFrame(data)
        
        # 1. Identify numeric columns that represent percentages
        format_dict = {}
        for col in df.columns:
            col_str = str(col).lower()
            if ('pct' in col_str or 'percent' in col_str or 'ratio' in col_str):
                # Only apply formatting if the column hasn't already been converted to a string
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Map the format string to the *future* Title Case column name
                    clean_name = str(col).replace("_", " ").title()
                    format_dict[clean_name] = "{:.2%}"
        
        # 2. Convert snake_case headers to Title Case
        df.columns = [str(c).replace("_", " ").title() for c in df.columns]
        
        # 3. Render the dataframe with the percentage styler
        st.dataframe(df.style.format(format_dict), width='stretch')
        
    elif "outlier_pct" in result:
        st.metric(
            "Outlier Rate", 
            f"{result.get('outlier_pct', 0):.2%}",
            help=f"{result.get('outlier_count', 0)} rows flagged"
        )
        contributors = result.get("top_contributing_columns", [])
        if contributors:
            df = pd.DataFrame(contributors)
            df.columns = [str(c).replace("_", " ").title() for c in df.columns]
            st.dataframe(df, width='stretch')
            
    elif not result.get("note"):
        st.caption("No data.")


def _render_multi_chart(result):
    for col, chart in result.get("data", {}).items():
        # Upgrade from st.caption to a proper Markdown header, and clean snake_case
        display_title = str(col).replace("_", " ").title()
        st.markdown(f"#### {display_title}")
        
        if chart["kind"] == "histogram":
            edges = chart["bin_edges"]
            midpoints = [f"{(edges[i] + edges[i + 1]) / 2:.1f}" for i in range(len(edges) - 1)]
            # Clean the Y-axis label as well
            st.bar_chart(pd.DataFrame({"Count": chart["counts"]}, index=midpoints))
            
        elif chart["kind"] == "bar":
            st.bar_chart(pd.DataFrame({"Count": chart["values"]}, index=chart["labels"]))


def _render_multi_chart_grouped(result):
    for col, chart in result.get("data", {}).items():
        display_title = str(col).replace("_", " ").title()
        st.markdown(f"#### {display_title}")
        
        if chart["kind"] == "grouped_histogram":
            edges = chart["bin_edges"]
            midpoints = [f"{(edges[i] + edges[i + 1]) / 2:.1f}" for i in range(len(edges) - 1)]
            
            df_chart = pd.DataFrame(chart["class_counts"], index=midpoints)
            # Ensure the legend (target classes) also uses Title Case if they are strings
            df_chart.columns = [str(c).replace("_", " ").title() for c in df_chart.columns]
            
            st.bar_chart(df_chart)

def _render_altair_boxplots(result):
    df_plot = pd.DataFrame(result.get("data", {}))
    for col in df_plot.columns:
        display_title = str(col).replace("_", " ").title()
        st.markdown(f"#### {display_title}")
        
        # Altair handles the IQR math and rendering automatically
        chart = alt.Chart(df_plot).mark_boxplot(extent=1.5).encode(
            x=alt.X(f"{col}:Q", title=None)
        ).properties(height=200)
        
        st.altair_chart(chart, use_container_width=True)

def _render_null_matrix(result):
    df_nulls = pd.DataFrame(result.get("data", {}))
    st.markdown("#### 🔳 Nullity Matrix")
    st.caption(f"Showing missing values (yellow) for {result.get('rows')} sampled rows.")
    
    # Dynamically scale the height based on how many columns exist
    fig_height = max(4, len(df_nulls.columns) * 0.3)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    
    # Transpose so columns are on the Y axis (mimicking the missingno library)
    ax.imshow(df_nulls.T, aspect='auto', cmap='viridis', interpolation='none')
    
    ax.set_yticks(range(len(df_nulls.columns)))
    ax.set_yticklabels([str(c).replace("_", " ").title() for c in df_nulls.columns])
    ax.set_xticks([]) # Hide arbitrary row numbers
    
    # Clean up the borders
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    st.pyplot(fig)

def _render_automl_results(result):
    task = result.get("task")
    leaderboard_data = result.get("leaderboard", [])
    feature_imp_data = result.get("feature_importance", [])
    
    st.markdown(f"### Model Performance Leaderboard ({task})")
    
    if leaderboard_data:
        df_leaderboard = pd.DataFrame(leaderboard_data)
        # Format metric columns beautifully using dynamic styling
        score_cols = [c for c in df_leaderboard.columns if c != "Model"]
        format_config = {col: "{:.4f}" for col in score_cols}
        st.dataframe(df_leaderboard.style.format(format_config), hide_index=True, use_container_width=True)
    
    st.divider()
    
    if feature_imp_data:
        st.markdown("### Tree Ensemble Feature Importance Insights")
        st.caption("Extracted from the Random Forest model execution bounds.")
        
        df_imp = pd.DataFrame(feature_imp_data)
        df_imp["Feature"] = df_imp["Feature"].apply(lambda x: str(x).replace("_", " ").title())
        
        st.bar_chart(df_imp.set_index("Feature")["Importance"])

