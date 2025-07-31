import streamlit as st
import plotly.graph_objects as go

def is_constraint_satisfied(constraint_name, actual, limit):
    if constraint_name.endswith("min"):
        return actual >= limit
    elif constraint_name.endswith("max"):
        return actual <= limit
    else:
        # Default to Min-type constraint
        return actual >= limit

def evaluation_printing(objective, constraints, constraints_met, problem):
    st.markdown("### Evaluation")
    st.metric("Objective", f"{objective:.2f}")

    constraints_string = ""
    for constraint_name, actual in constraints.items():
        if constraint_name in problem["constraints"]:
            limit = problem["constraints"][constraint_name]
            satisfied = is_constraint_satisfied(constraint_name, actual, limit)

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=[limit],
                y=[constraint_name],
                name="Limit",
                orientation='h',
                marker=dict(color='lightgray'),
                hoverinfo='x'
            ))

            fig.add_trace(go.Bar(
                x=[actual],
                y=[constraint_name],
                name="Actual",
                orientation='h',
                marker=dict(color='green' if satisfied else 'red'),
                hovertemplate=f"{constraint_name}: {actual:.2f} (limit: {limit})<extra></extra>"
            ))

            fig.update_layout(
                barmode='overlay',
                title=f"{constraint_name}",
                xaxis=dict(title='Value'),
                height=100,
                margin=dict(t=30, b=30, l=50, r=20),
                showlegend=False
            )

            col = columns[idx % 4]
            col.plotly_chart(fig, use_container_width=True)
        else:
            constraints_string += f"{actual}  |  "

    st.write(constraints_string)

    if constraints_met:
        st.success("Constraints met!")
    else:
        st.warning("Constraints not satisfied.")