
import streamlit as st
from problems.problem_collection import solution_evaluation, diet_problem, food_distribution_problem
from utils.evaluate_input import evaluate_and_generate_code
from login_page import login

# Page config
st.set_page_config(page_title='Using GenAI in practice', page_icon='', layout = 'wide')

# Display LTP logo
st.image(image= "images/Asset 6.png", caption = "Powered by", width = 100)

# Session state initialization
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "selected_page" not in st.session_state:
    st.session_state["selected_page"] = "Home"

# Log in page
if not st.session_state["logged_in"]:
    login()

else:
    st.sidebar.title("Prescriptive Analytics Tool")
    st.session_state["selected_page"] = st.sidebar.radio("Choose what to do:", ["Home", "Diet problem", "Food distribution problem"])
    if st.session_state["selected_page"] == "Home":
        # Introduction to the tool
        st.title("Welcome to the Prescriptive Analytics Learning Tool")

        st.markdown("""
            ### 🎯 Goal
            This interactive tool is designed to help students and professionals **learn prescriptive analytics** by:
            - Understanding optimization problems
            - Trying to solve them manually
            - Defining variables, constraints, and objectives
            - Comparing their approach with the **optimal solution** via Python + PuLP

            ### 🧩 How it works
            1. Pick a problem from the sidebar
            2. Try to solve it manually (without optimization)
            3. Describe the decision model in words (variables, constraints, objective)
            4. Let the tool generate the optimization code
            5. Compare your result with the optimal solution

            ---
            👈 Start by selecting **"Solve a Problem"** in the sidebar!
            """)
    else:
        if st.session_state["selected_page"] == "Diet problem":
            PROBLEM = diet_problem()
        elif st.session_state["selected_page"] == "Food distribution problem":
            PROBLEM = food_distribution_problem()
        else:
            PROBLEM = diet_problem()

        # Page title and problem description
        st.title("Prescriptive AI - " + PROBLEM["title"])
        st.header('Problem description', divider='rainbow')
        st.markdown(PROBLEM["description"])
        if "dataframe" in PROBLEM:
            st.dataframe(PROBLEM["dataframe"])

        # Allow user to play with solution for the problem and check results
        st.header('Autonomous solution', divider='rainbow')
        with st.form("manual_guess"):
            user_input = {}
            var_items = list(PROBLEM["vars"]["vars"].items())
            col1, col2 = st.columns(2)

            for i, (var_name, _) in enumerate(var_items):
                col = col1 if i % 2 == 0 else col2
                with col:
                    user_input[var_name] = st.number_input(f"Units of {var_name}", min_value=0.0, step=0.1)

            submitted = st.form_submit_button("Evaluate")

            if submitted:
                constraints_evaluation = {}
                objective_evaluation = 0
                constraints_met = False
                objective_evaluation, constraints_evaluation, constraints_met = solution_evaluation(PROBLEM, user_input)

                st.metric("Objective", f"{objective_evaluation:.2f}")

                constraints_string = ""
                for constraint_name, actual in constraints_evaluation.items():
                    if constraint_name in PROBLEM["constraints"]:
                        limit = PROBLEM["constraints"][constraint_name]
                        constraints_string += f"{constraint_name}: {actual} (limit: {limit})  |  "
                    else:
                        constraints_string += f"{actual}  |  "

                st.write(constraints_string)

                if constraints_met:
                    st.success("Constraints met!")
                else:
                    st.warning("Constraints not satisfied.")

        # Ask the user what are the key characteristics of the problem
        st.header('Optimization model construction', divider='rainbow')
        with st.form("model_entry"):
            decision_vars = st.text_area("What are the decision variables of the problem?")
            constraints = st.text_area("What are the constraints of the problem?")
            objective = st.text_area("What is the objective function of the problem?")

            submitted_model = st.form_submit_button("Submit Model")

            # Evaluate the user responses and generate a solution for the problem
            if submitted_model:
                success, feedback, pulp_code = evaluate_and_generate_code(
                    decision_vars, constraints, objective, PROBLEM["description"], PROBLEM["structured_data"]
                )

                st.markdown(feedback)

                if success:
                    st.subheader("Generated Optimization Code")
                    st.code(pulp_code, language='python')

                    local_vars = {}
                    try:
                        exec(code_object, local_vars, local_vars)
                        result = local_vars.get("result")
                        if result is not None:
                            st.header('Optimization model assessment', divider='rainbow')
                            for var, qty in result["solution"].items():
                                st.write(f"**{var.title()}**: {qty:.2f} units")
                            st.metric("Optimized goal", f"${result['objective']:.2f}")
                        else:
                            st.error("Optimization result not found. Check if 'result' is assigned in your code.")
                    except Exception as e:
                        st.error(f"Error running generated code: {e}")
                        #st.code(pulp_code, language='python')
                else:
                    st.warning("Try to correct your model and resubmit.")