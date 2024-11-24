import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from config import page_titles
from sidebar import sidebar_config

def introduction_text():
    st.header('**What can this app do?**')
    with st.expander('**Click to see explanation**', expanded=False):
        st.write('This app allow users to build a machine learning (ML) model with an end-to-end workflow simple steps:\n')
        for i in range(len(page_titles)-1):
            st.write(f'- {page_titles[i+1]}\n')

    st.header('**How to use the app?**')
    with st.expander('**Click to see explanation**', expanded=False):
        st.write('To engage with the app, you will be able to use the sidebar to make choices that will help prepare and train the machine learning model. Some examples of choices are:\n1. Upload a data set\n2. Select the data imputation methods\n3. Adjust the model training and parameters\nYou will be able to go back and forth to understand the impact of different choices on the results')

    st.header('Data Loading', divider='rainbow')
    if not st.session_state.uploaded:
        st.write('Upload a dataset on the sidebar')
    else:
        st.write('This is your dataset:')
        df = st.session_state.df_original
        st.dataframe(df, height = 300)
        st.write('The last column of the dataset will be considered your target variable') # Review at the end
        st.write('These are the data types identified for your dataset:')
        st.write(df.dtypes)