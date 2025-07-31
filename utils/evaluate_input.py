
import streamlit as st
from openai import AzureOpenAI
import os

client = AzureOpenAI(
        api_key=st.secrets["AZURE_OPENAI_KEY"],
        azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
        api_version=st.secrets["AZURE_OPENAI_API_VERSION"]
    )

def extract_text_and_code(content):
    parts = content.split("```python")
    text_before_code = parts[0].strip()

    code_block = ""
    if len(parts) > 1 and "```" in parts[1]:
        code_block = parts[1].split("```")[0].strip()

    return text_before_code, code_block

def evaluate_and_generate_code(user_vars, user_constraints, user_objective, problem_description, structured_data):
    prompt = f"""
            Problem given to the studen:
            {problem_description}

            Structured data for the problem:
            {structured_data}

            The student provided:
            - Decision variables: {user_vars}
            - Constraints: {user_constraints}
            - Objective: {user_objective}

            1. Determine if these are the correct answers to the three questions.
            2. Give an overall assessment of the answer.
            3. If you consider all the answers correct, generate Python code using PuLP that defines and solves the problem. Explain the though-process and the code in a user-friendly manner. The optimization model must use Mixed Integer Linear Programming (MILP). Introduce binary variables when needed to ensure linearity. Use big-M constraint to link continuous and binary variables.
            4. If you consider all the answers correct, return the code inside a Python triple-quoted string.
            5. If code is returned, it must define a variable named `result` as a dictionary like:
                result = {{
                    "solution": {{"var1": ..., "var2": ..., "var3": ...}},
                    "objective": ..., 
                    "status": ...
                    }}
                    Do not use print(). Store everything in `result`.

            Return the code inside triple backticks.
            """
    #"gpt-4o-mini", "gpt-4", "claude-3-5-sonnet", "claude-3-7-sonnet"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[  
                {"role": "system", "content": "You are an expert in mathematical optimization, experienced at using PuLP library to model optimization problems in Python. Help solving the problems, explaining the thought-process to an audience with little knowledge about the topic"},
                {"role": "user", "content": prompt}
            ],
        temperature=0,
    )

    code_block = None
    core_message = None
    for choice in response.choices:
        content = choice.message.content
        if "```python" in content:
            core_message, code_block = extract_text_and_code(content)
            break

    if not code_block:
        return False, choice.message.content, None

    return True, core_message, code_block
