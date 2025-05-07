import streamlit as st
import pandas as pd
import numpy as np
from config import page_titles
from login_page import login
from sidebar import sidebar_config
from page_body import introduction_text, exploratory_data_analysis, data_preparation, model_training, result_analysis, model_interpretation, exercise_summary

# Navigation function with forced rerun
def change_page(delta):
    st.session_state.page = max(0, min(len(page_titles) - 1, st.session_state.page + delta))
    st.rerun()  # Force immediate rerun to reflect the updated page state

# Initialize session state variables
if "page" not in st.session_state:
    st.session_state.page = -1

if "uploaded" not in st.session_state:
    st.session_state.uploaded = False

if "treated" not in st.session_state:
    st.session_state.treated = False

if "trained" not in st.session_state:
    st.session_state.trained = False

if "predict_output" not in st.session_state:
    st.session_state.predict_output = False

if "run_id" not in st.session_state:
    st.session_state.run_id = 0

for k, v in st.session_state.items():
    if k != 'page' and 'next' not in k and 'prev' not in k and 'bot_restart' not in k and 'top_restart' not in k:
        st.session_state[k] = v

current_page = st.session_state.page

# Page config
st.set_page_config(page_title='Building a ML model', page_icon='', layout = 'wide')

# Display LTP logo
st.image(image= "images/Asset 6.png", caption = "Powered by", width = 100)

if current_page > 0:
    if st.button("Restart", use_container_width=True, key=f"top_restart_{current_page}"):
        st.session_state.page = 0
        st.session_state.uploaded = False
        st.session_state.treated = False
        st.session_state.trained = False
        st.session_state.predict_output = False
        st.session_state.df_original = pd.DataFrame()
        st.session_state.df_treated = pd.DataFrame()
        st.rerun()

# Display title of the page
st.title(page_titles[current_page], anchor='title')

sidebar_config(current_page)

if current_page == -1:
    # Session state check
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if st.session_state["logged_in"]:
        st.session_state.page = 0
    else:
        login()

if current_page == 0:
    introduction_text()

if current_page == 1:
    if not st.session_state.uploaded:
        st.write('Go back to the previous page and reupload your dataset')
    else:
        exploratory_data_analysis()

if current_page == 2:
    if not st.session_state.uploaded:
        st.write('Go back to the beginning and reupload your dataset')
    else:
        data_preparation()

if current_page == 3:
    if not st.session_state.treated:
        st.write('Go back to the previous page and prepare your dataset')
    else:
        model_training()

if current_page == 4:
    if not st.session_state.trained:
        st.write('Go back to the previous page and train your model')
    else:
        result_analysis()

if current_page == 5:
    if not st.session_state.trained:
        st.write('Go back to the model training page and train your model')
    else:
        model_interpretation()

if current_page == 6:
    if not st.session_state.trained:
        st.write('Go back to the model training page and train your model')
    else:
        exercise_summary()

# Display buttons at the end to navigate between pages
if current_page == 0:
    left, right = st.columns(2)
    if right.button("Next", use_container_width=True, key=f"next_{current_page}"):
        change_page(1)

elif 0 < current_page < len(page_titles)-1:
    left, right = st.columns(2)
    if left.button("Previous", use_container_width=True, key=f"prev_{current_page}"):
        change_page(-1)
    if right.button("Next", use_container_width=True, key=f"next_{current_page}"):
        change_page(1)

elif current_page == len(page_titles)-1:
    left, right = st.columns(2)
    if left.button("Previous", use_container_width=True, key=f"prev_{current_page}"):
        change_page(-1)
# Restart if needed
else:
    st.session_state.page = 0

if current_page > 0:
    if st.button("Restart", use_container_width=True, key=f"bot_restart_{current_page}"):
        st.session_state.page = 0
        st.session_state.uploaded = False
        st.session_state.treated = False
        st.session_state.trained = False
        st.session_state.predict_output = False
        st.session_state.df_original = pd.DataFrame()
        st.session_state.df_treated = pd.DataFrame()
        st.rerun()
# Debug
df = pd.read_csv('Customer_Churn.csv', sep=';', index_col=False, decimal='.')  
df
df.dtypes
# col = df.columns[13]
for col in df.columns:
    if len(pd.to_numeric(df[col], errors='coerce').dropna().unique()) >= 0.5 * len(df[col]):
        df[col] = pd.to_numeric(df[col], errors='coerce')
df.dtypes
df
categorical_columns = df.select_dtypes(include=['object', 'category']).columns
encoder = OneHotEncoder(sparse_output=False, drop='if_binary')

# Apply one-hot encoding
encoded_data = encoder.fit_transform(df[categorical_columns])

# Create a DataFrame for the encoded columns
encoded_df = pd.DataFrame(
    encoded_data,
    columns=encoder.get_feature_names_out(categorical_columns)
)

# Combine with the original DataFrame (excluding the original categorical columns)
# df = pd.concat([df.drop(columns=categorical_columns), encoded_df], axis=1)
# to_predict = 'ChurnNext6Months_Yes'
# input_variables=list(set(df.columns) - {to_predict})
# y = df[to_predict]
# x = df[input_variables]
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=20/100, random_state=42)
# ml_mod = LogisticRegression()
# ml_mod.fit(x_train, y_train)
# y_train_pred = ml_mod.predict(x_train)
# y_test_pred = ml_mod.predict(x_test)
# conf_matrix = confusion_matrix(y_test, y_test_pred)

# x_train_sm = sm.add_constant(x_train)  # Add intercept term
# logit_model = sm.Logit(y_train, x_train_sm).fit()

# coeffs_df = pd.DataFrame({
#             "Feature": ["Intercept"] + list(x_train.columns),
#             "Coefficient": [logit_model.params[0]] + list(logit_model.params[1:]),
#             "Odds Ratio": np.exp([logit_model.params[0]] + list(logit_model.params[1:])),
#             "P-Value": [logit_model.pvalues[0]] + list(logit_model.pvalues[1:]),
#             "95% CI Lower": logit_model.conf_int().iloc[:, 0],
#             "95% CI Upper": logit_model.conf_int().iloc[:, 1]
#         })
# coeffs_df = coeffs_df.sort_values("Odds Ratio", key=abs, ascending=False)