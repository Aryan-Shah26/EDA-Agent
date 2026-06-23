import os
import json
import pandas as pd
from src.ingestion.data_context import DataContextObject

PROMPT_TEMPLATE = """You are the elite AI Data Co-Pilot. The user has asked a question about their dataset.
You have a Pandas DataFrame loaded as `df`.

Columns, types, and domain context:
{schema}

Question: {question}

--- STRICT BEHAVIORAL RULES ---

1. OBJECTIVE ALIGNMENT (Scalar vs. Visual):
- SCALAR QUERIES: If the user asks for a number, percentage, count, average, or list, you MUST calculate it and assign it to a variable named `result`. You must NEVER generate a plot, chart, or graph.
- VISUAL QUERIES: ONLY if the user explicitly asks to "plot", "chart", "graph", or "visualize", use `import plotly.express as px` and assign the figure to a variable named `fig`. Do NOT use `fig.show()`.

2. SAFE STRING FILTERING:
Users miscapitalize things. Do NOT use standard exact matching (`df['col'] == 'Apple'`).
Instead, normalize both sides to lowercase for safe exact matching:
`df[df['col'].astype(str).str.lower() == 'target_value'.lower()]`
Only use `.str.contains()` if asked for a partial match.

3. PANDAS 2.0 VALUE_COUNTS COMPATIBILITY:
When plotting value counts, NEVER assume the columns will be named 'index' after running `.reset_index()`. 
ALWAYS explicitly rename the columns:
```python
counts_df = df['column_name'].value_counts().reset_index()
counts_df.columns = ['value', 'count']
fig = px.bar(counts_df, x='value', y='count')

THE "EMPTY RESULT" PROTOCOL:
If you filter the DataFrame and the resulting DataFrame is empty, assign a polite string explaining this to the result variable. DO NOT hallucinate numbers.

OUTPUT FORMAT:
Write EXACTLY ONE Python code block. Do not write conversational text outside the block.

Python
# Your code here
{history}
"""

def run_with_self_correction(question: str, dco: DataContextObject, llm_fn, max_retries=3):

    # Build an enriched schema info payload for the prompt
    schema_details = {}
    for col, prof in dco.columns.items():
        desc = getattr(prof, 'description', 'No context provided.')
        schema_details[col] = {
            "type": str(prof.dtype),
            "context": desc
        }
    schema_info = json.dumps(schema_details, indent=2)
    
    # Run against the reservoir sample to prevent OOM crashes on huge datasets
    if not dco.reservoir_sample_path or not os.path.exists(dco.reservoir_sample_path):
        yield {"status": "failed", "error": "No data sample available for coding."}
        return
        
    df = pd.read_parquet(dco.reservoir_sample_path)
    
    history_text = ""
    last_code = ""
    
    for attempt in range(max_retries):
        # 1. Ask LLM to generate code
        prompt = PROMPT_TEMPLATE.format(schema=schema_info, question=question, history=history_text)
        response = llm_fn(prompt).strip()
        
        # 2. Extract code block safely
        code = response
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()
        
        code = code.replace("fig.show()", "")
        last_code = code
        
        # 3. Sandbox Execution Environment
        local_vars = {"df": df, "pd": pd}
        
        try:
            # Execute the code in our restricted local namespace
            exec(code, {}, local_vars)
            
            # Look for expected outputs
            if "fig" in local_vars:
                yield {"status": "success", "type": "chart", "data": local_vars["fig"], "code": code, "attempts": attempt + 1}
                return
            elif "result" in local_vars:
                yield {"status": "success", "type": "text", "data": str(local_vars["result"]), "code": code, "attempts": attempt + 1}
                return
            else:
                raise ValueError("Code executed successfully but did not assign output to 'fig' or 'result'.")
                
        except Exception as e:
            # 4. SELF CORRECTION: Catch error and append to history for the next loop!
            error_msg = str(e)
            history_text += f"\n\n--- ATTEMPT {attempt + 1} FAILED ---\nCode:\n{code}\nError:\n{error_msg}\nPlease fix the error and try again."
            
            # Yield a status update to the UI
            yield {"status": "retrying", "attempt": attempt + 1, "error": error_msg, "code": code}
            
    yield {"status": "failed", "error": "Max retries reached. The agent could not fix the code.", "code": last_code}