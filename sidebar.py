import streamlit as st
import pandas as pd

def sidebar_config(i):
    with st.sidebar:
        if i == 0:
            # Load data
            st.header('Input data')

            col_sep = st.selectbox(
                'What is the column separator of your file:',
                [',',';']
            )
            dec_id = st.selectbox(
                'What is the decimal point character:',
                ['.',',']
            )

            uploaded_file = st.file_uploader("Upload the training set", type=["csv"])
            if uploaded_file is not None:
                st.session_state.uploaded  = True
                st.session_state.df_original = pd.read_csv(uploaded_file, sep=col_sep, index_col=False, decimal=dec_id)

            #Revise in the end
            #gam_data = st.toggle('Export Predictions')
            #if gam_data is True:
            #    gam_file = st.file_uploader("Upload the data to predict", type=["csv"])
            #    if gam_file is not None:
            #        df_gam = pd.read_csv(gam_file, sep=';', index_col=False)

        elif i == 1:
            # Select variables to analyze in detail
            st.header('Variable selection')
            var_1 = st.selectbox(
                'Select a variable to analyze in detail:',
                st.session_state.df_original.columns
            )
            st.session_state.var_1 = var_1

            var_2 = st.selectbox(
                'Select a second variable to analyze in detail:',
                list(set(st.session_state.df_original.columns) - {var_1})
            )
            st.session_state.var_2 = var_2

        elif i == 2:
            st.header('Data preparation')
            st.write('Categorical data')
            categorical_treat = st.selectbox(
                'How to treat categorical data:',
                ['Remove columns', 'Label encoding', 'One-hot encoding'] #Add target encoding in the future
            )
            st.session_state.categorical_treat = categorical_treat

            st.write('Missing values treatment')
            missing_treat = st.selectbox(
                'How to treat missing values:',
                ['Remove observation', 'Imputation: mean', 'Imputation: median'] 
            )
            st.session_state.missing_treat = missing_treat

            #Add option of how to find outliers
            st.write('Outlier treatment')
            outlier_treat = st.selectbox(
                'How to treat outlier values:',
                ['Keep as-is', 'Remove observation', 'Imputation: mean', 'Imputation: median'] 
            )
            st.session_state.outlier_treat = outlier_treat

        elif i == 3:

            st.header('Problem Type')
            problem_type = st.sidebar.radio('Choose problem type:', ['Regression', 'Classification'])
            st.session_state.problem_type = problem_type

            st.header('Training Parameters')
            parameter_split_size = st.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
            st.session_state.parameter_split_size = parameter_split_size

            st.header('Model Parameters')
            st.subheader('Machine learning model')
            if problem_type == 'Regression':
                model_to_use = st.sidebar.radio('Choose model:', ['Linear regression', 'Random forest', 'Gradiant boosting machines'])
            else:
                model_to_use = st.sidebar.radio('Choose model:', ['Logistic regression', 'Random forest', 'Gradiant boosting machines'])
            st.session_state.model_to_use = model_to_use

            st.header('Learning Parameters')
            st.session_state.parameter_n_estimators = st.slider('Number of estimators (n_estimators)', 5, 50, 5, 5)
            #parameter_max_features = st.select_slider('Max features (max_features)', options=['all', 'sqrt', 'log2'])
            #parameter_min_samples_split = st.slider('Minimum number of samples required to split an internal node (min_samples_split)', 2, 10, 2, 1)
            #parameter_min_samples_leaf = st.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

            st.header('Other Parameters')
            if problem_type == 'Regression':
                parameter_criterion = st.sidebar.radio('Performance measure (criterion)', ['squared_error', 'absolute_error', 'poisson', 'firedman_mse', ])
            else:
                parameter_criterion = st.sidebar.radio('Performance measure (criterion)', ['gini', 'entropy', 'log_loss'])
            st.session_state.parameter_criterion = parameter_criterion
            st.session_state.parameter_random_state = st.slider('Seed number (random_state)', 0, 1000, 42, 1)
            #parameter_bootstrap = st.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
            #parameter_oob_score = st.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])

        elif i == 4:
            st.header('Result Parameters')
            st.subheader('Metric analysis')
            if problem_type == 'Regression':
                st.session_stateme_analysis = st.sidebar.radio('Compare Mean Error:', ['Yes', 'No'])
                st.session_statemse_analysis = st.sidebar.radio('Compare Mean Square Error:', ['Yes', 'No'])
                st.session_statemape_analysis = st.sidebar.radio('Compare Mean Absolute Percentual Error:', ['Yes', 'No'])
            else:
                st.session_stateconf_analysis = st.sidebar.radio('Analyze confusion matrix:', ['Yes', 'No'])
                st.session_stateaccuracy_analysis = st.sidebar.radio('Compare accuracy:', ['Yes', 'No'])
                st.session_stateprecision_analysis = st.sidebar.radio('Compare precision:', ['Yes', 'No'])
                st.session_staterecall_analysis = st.sidebar.radio('Compare recall:', ['Yes', 'No'])
        
        elif i == 5:
            st.header('Model interpretation')
            st.subheader('Type of analysis')
            