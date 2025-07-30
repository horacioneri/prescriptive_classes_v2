
from openai import AzureOpenAI
import os

client = AzureOpenAI(
        api_key=st.secrets["AZURE_OPENAI_KEY"],
        azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
        api_version=st.secrets["AZURE_OPENAI_API_VERSION"]
    )

def evaluate_and_generate_code(user_vars, user_constraints, user_objective, food_data, constraints_data):
    prompt = f"""
            Problem:
            Choose quantities of foods to meet nutritional needs at the lowest cost.

            Available foods:
            {food_data}

            Constraints:
            {constraints_data}

            The student provided:
            - Decision variables: {user_vars}
            - Constraints: {user_constraints}
            - Objective: {user_objective}

            1. Determine if this are the correct answers to the three questions.
            2. If they are correct, generate Python code using PuLP that defines and solves the problem. Explain the though-process and the code in a user-friendly manner.
            3. Return only the code inside a Python triple-quoted string (no explanation).

            Only return the code inside triple backticks.
            """
    #"gpt-4o-mini", "gpt-4", "claude-3-5-sonnet", "claude-3-7-sonnet"
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[  
                {"role": "system", "content": "You are an expert in mathematical optimization, experienced at using PuLP library to model optimization problems in Python. Help solving the problems, explaining the thought-process to an audience with little knowledge about the topic"},
                {"role": "user", "content": prompt}
            ],
        temperature=0,
    )

    code_block = None
    for choice in response.choices:
        content = choice.message.content
        if "```python" in content:
            code_block = content.split("```python")[1].split("```")[0].strip()
            break

    if not code_block:
        return False, "❌ Model evaluation failed or was incorrect.", None

    return True, "✅ Model correct and code generated.", code_block
