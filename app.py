
import streamlit as st
from problems.problem_collection import diet_problem, solution_evaluation
from utils.evaluate_input import evaluate_and_generate_code
from login_page import login

# Page config
st.set_page_config(page_title='Using GenAI in practice', page_icon='', layout = 'wide')

# Display LTP logo
st.image(image= "images/Asset 6.png", caption = "Powered by", width = 100)
#st.info("This app demonstrates the use of GenAI and agents for document understanding, Q&A, and dynamic visualization generation in practical sessions. It is a simplified version designed for instructional purposes.")

# Session state initialization
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# Log in page
if not st.session_state["logged_in"]:
    login()

else:
    PROBLEM = diet_problem()
    st.title("Prescriptive AI - " + PROBLEM["title"])
    st.header('Problem description', divider='rainbow')
    st.markdown(PROBLEM["description"])

    st.header('Autonomous solution', divider='rainbow')
    with st.form("manual_guess"):
        user_input = {}
        for var_name, _ in PROBLEM["vars"]["vars"].items():
            user_input[var_name] = st.number_input(f"Units of {var_name}", min_value=0.0, step=0.1)
        
        submitted = st.form_submit_button("Evaluate")

        if submitted:
            constraints_evaluation = {}
            objective_evaluation = 0
            constraints_met = False
            objective_evaluation, constraints_evaluation, constraints_met = solution_evaluation(PROBLEM, user_input)

            st.metric("Objective", f"{objective_evaluation:.2f}")

            constraints_string = ""
            for constraint_name, value in PROBLEM["constraints"].items():
                actual = constraints_evaluation.get(constraint_name, "N/A")
                constraints_string += f"{constraint_name}: {actual} (limit: {value})  |  "

            st.write(constraints_string)

            if constraints_met:
                st.success("Constraints met!")
            else:
                st.warning("Constraints not satisfied.")

    st.header('Optimization model construction', divider='rainbow')
    with st.form("model_entry"):
        decision_vars = st.text_area("What are the decision variables of the problem?")
        constraints = st.text_area("What are the constraints of the problem?")
        objective = st.text_area("What is the objective function of the problem?")

        submitted_model = st.form_submit_button("Submit Model")
        if submitted_model:
            success, feedback, pulp_code = evaluate_and_generate_code(
                decision_vars, constraints, objective, PROBLEM["description"]
            )

            st.markdown(feedback)

            if success:
                st.subheader("Generated Optimization Code")
                st.code(pulp_code, language='python')

                local_vars = {}
                try:
                    exec(pulp_code, {}, local_vars)
                    if "result" in local_vars:
                        result = local_vars["result"]
                        st.header('Optimization model assessment', divider='rainbow')
                        for var, qty in result["solution"].items():
                            st.write(f"**{var.title()}**: {qty:.2f} units")
                        st.metric("Optimized goal", f"${result['objective']:.2f}")
                    else:
                        st.error("Optimization result not found in generated code.")
                except Exception as e:
                    st.error(f"Error running generated code: {e}")
            else:
                st.warning("Try to correct your model and resubmit.")