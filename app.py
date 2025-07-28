
import streamlit as st
from problems.diet_problem import PROBLEM
from utils.optimizer import solve_diet_problem
from utils.evaluate_input import evaluate_and_generate_code

st.set_page_config(page_title="Prescriptive Analytics Playground", layout="centered")
st.title(PROBLEM["title"])
st.markdown(PROBLEM["description"])

st.header("🎮 Try to solve manually")
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

st.header("💬 Define Your Model")
with st.form("model_entry"):
    decision_vars = st.text_input("List your decision variables (comma separated)")
    constraints = st.text_area("Write down the constraints")
    objective = st.text_input("Describe the objective function")

    submitted_model = st.form_submit_button("Submit Model")
    if submitted_model:
        var_list = [v.strip().lower() for v in decision_vars.split(",")]

        success, feedback, pulp_code = evaluate_and_generate_code(
            decision_vars, constraints, objective,
            food_data=PROBLEM["foods"], constraints_data=PROBLEM["constraints"]
        )

        st.markdown(feedback)

        if success:
            st.subheader("🔧 Generated Optimization Code")
            st.code(pulp_code, language='python')

            local_vars = {}
            try:
                exec(pulp_code, {}, local_vars)
                if "result" in local_vars:
                    result = local_vars["result"]
                    st.subheader("📈 Optimized Solution")
                    for food, qty in result["solution"].items():
                        st.write(f"**{food.title()}**: {qty:.2f} units")
                    st.metric("Optimized Cost", f"${result['objective']:.2f}")
                else:
                    st.error("Optimization result not found in generated code.")
            except Exception as e:
                st.error(f"Error running generated code: {e}")
        else:
            st.warning("Try to correct your model and resubmit.")
