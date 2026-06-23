import re
import json

def parse_domain_dictionary(text_content: str, columns: list) -> dict:
    """
    Uses a fast LLM pass to map an uploaded text data dictionary 
    to the specific columns present in our dataset.
    """
    from src.agent.llm_router import get_llm
    
    prompt = f"""
    You are given a raw text data dictionary/domain context document and a list of columns from a dataset.
    Match each column to its definition or relevant context found in the text.
    
    Columns in dataset: {columns}
    
    Domain Text Document:
    \"\"\"
    {text_content}
    \"\"\"
    
    Return a valid JSON object where keys are the EXACT column names and values are short, clear summaries of their business meaning or domain rules extracted from the text. Do not include columns that are not mentioned.
    Format:
    {{
        "column_name": "definition and rules"
    }}
    """
    
    try:
        fast_llm = get_llm("fast")
        response = fast_llm.invoke(prompt).content.strip()
        
        # Clean up markdown code blocks if present
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()
            
        return json.loads(response)
    except Exception as e:
        print(f"Error parsing domain dictionary: {e}")
        return {}