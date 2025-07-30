
import streamlit as st
from problems.diet_problem import PROBLEM
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
    st.title("Prescriptive AI - " + PROBLEM["title"])
    st.header('Problem description', divider='rainbow')
    st.markdown(PROBLEM["description"])

    st.header('Autonomous solution', divider='rainbow')
    with st.form("manual_guess"):
        chicken = st.number_input("Units of Chicken", min_value=0.0, step=0.1)
        rice = st.number_input("Units of Rice", min_value=0.0, step=0.1)
        broccoli = st.number_input("Units of Broccoli", min_value=0.0, step=0.1)
        submitted = st.form_submit_button("Evaluate")

        if submitted:
            total_cost = chicken * 2 + rice * 0.5 + broccoli * 1.0
            protein = chicken*30 + rice*3 + broccoli*2
            carbs = rice*30 + broccoli*10
            fat = chicken*4 + broccoli*1

            st.metric("Total Cost", f"${total_cost:.2f}")
            st.write(f"Protein: {protein}g | Carbs: {carbs}g | Fat: {fat}g")

            if protein >= 50 and carbs <= 70 and fat <= 10:
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