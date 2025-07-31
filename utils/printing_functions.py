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
    columns = st.columns(4)
    for idx, (constraint_name, actual) in enumerate(constraints.items()):
        if constraint_name in problem["constraints"]:
            limit = problem["constraints"][constraint_name]
            satisfied = is_constraint_satisfied(constraint_name, actual, limit)

            fig = go.Figure()

            # Main bars
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
                marker=dict(color='lightgreen' if satisfied else 'lightred'),
                hovertemplate=f"{constraint_name}: {actual:.2f} (limit: {limit})<extra></extra>"
            ))

            # Add a line showing the limit, if actual > limit (for max) or actual < limit (for min)
            show_limit_line = (
                (constraint_name.endswith("max") and actual > limit) or
                (constraint_name.endswith("min") and actual < limit) or
                (not constraint_name.endswith(("max", "min")) and actual < limit)  # default: Min
            )

            if show_limit_line:
                fig.add_shape(
                    type="line",
                    x0=limit,
                    x1=limit,
                    y0=-0.5,
                    y1=0.5,
                    line=dict(color="black", dash="dash"),
                )

            # Layout
            fig.update_layout(
                barmode='overlay',
                xaxis=dict(title=''),
                height=120,
                margin=dict(t=25, b=25, l=25, r=10),
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