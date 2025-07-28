
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def evaluate_and_generate_code(user_vars, user_constraints, user_objective, food_data, constraints_data):
    prompt = f"""
You are a teaching assistant for an optimization course. A student has entered their modeling attempt for the following diet problem:

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

1. Determine if this is a correct linear program model for the problem.
2. If it's correct, generate Python code using PuLP that defines and solves the problem.
3. Return only the code inside a Python triple-quoted string (no explanation).

Only return the code inside triple backticks.
"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
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
