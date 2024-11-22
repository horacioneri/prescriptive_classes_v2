import streamlit as st

def sidebar_config(i):
    with st.sidebar:
        if i == 1:
            # Load data
            st.header('Input data')

            uploaded_file = st.file_uploader("Upload the training set", type=["csv"])
            if uploaded_file is not None:
                st.session_state.uploaded  = True
                st.session_state.df_original = pd.read_csv(uploaded_file, sep=';', index_col=False)
                df = st.session_state.df_original

            #Revise in the end
            gam_data = st.toggle('Export Predictions')
            if gam_data is True:
                gam_file = st.file_uploader("Upload the data to predict", type=["csv"])
                if gam_file is not None:
                    df_gam = pd.read_csv(gam_file, sep=';', index_col=False)

            # Load data
            st.header('Data preparation')
            st.write('Categorical data')
            categorical_treat = st.selectbox(
                'How to treat categorical data:',
                ['Remove column', 'Label encoding', 'One-hot encoding'] #Add target encoding in the future
            )

            st.write('Missing values treatment')
            missing_treat = st.selectbox(
                'How to treat missing values:',
                ['Remove observation', 'Imputation: mean', 'Imputation: median'] 
            )

            #Add option of how to find outliers
            st.write('Outlier treatment')
            missing_treat = st.selectbox(
                'How to treat outlier values:',
                ['Keep as-is', 'Remove observation', 'Imputation: mean', 'Imputation: median'] 
            )
        
        elif i == 2:
            # Select variables to analyze in detail
            st.header('Variable selection')
            var_1 = st.selectbox(
                'Select a variable to analyze in detail:',
                df_treated.columns
            )

            var_2 = st.selectbox(
                'Select a second variable to analyze in detail:',
                df_treated.columns
            )

        elif i == 3:

            st.header('Problem Type')
            problem_type = st.sidebar.radio('Choose model type:', ['Regression', 'Classification'])

            st.header('Training Parameters')
            parameter_split_size = st.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

            st.header('Model Parameters')
            st.subheader('Problem Type')
            problem_type = st.sidebar.radio('Choose model type:', ['Regression', 'Classification'])
            st.subheader('Machine learning model')
            if problem_type == 'Regression':
                model_to_use = st.sidebar.radio('Choose model type:', ['Linear regression', 'Random forest', 'Gradiant boosting machines'])
            else:
                model_to_use = st.sidebar.radio('Choose model type:', ['Logistic regression', 'Random forest', 'Gradiant boosting machines'])

            st.header('Learning Parameters')
            parameter_n_estimators = st.slider('Number of estimators (n_estimators)', 5, 50, 5, 5)
            #parameter_max_features = st.select_slider('Max features (max_features)', options=['all', 'sqrt', 'log2'])
            #parameter_min_samples_split = st.slider('Minimum number of samples required to split an internal node (min_samples_split)', 2, 10, 2, 1)
            #parameter_min_samples_leaf = st.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

            st.header('Other Parameters')
            if problem_type == 'Regression':
                parameter_criterion = st.select_slider('Performance measure (criterion)', options=['squared_error', 'absolute_error', 'poisson', 'firedman_mse', ])
            else:
                parameter_criterion = st.select_slider('Performance measure (criterion)', options=['gini', 'entropy', 'log_loss'])
            parameter_random_state = st.slider('Seed number (random_state)', 0, 1000, 42, 1)
            #parameter_bootstrap = st.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
            #parameter_oob_score = st.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])

        elif i == 4:
            st.header('Result Parameters')
            st.subheader('Metric analysis')
            if problem_type == 'Regression':
                me_analysis = st.sidebar.radio('Compare Mean Error:', ['Yes', 'No'])
                mse_analysis = st.sidebar.radio('Compare Mean Square Error:', ['Yes', 'No'])
                mape_analysis = st.sidebar.radio('Compare Mean Absolute Percentual Error:', ['Yes', 'No'])
            else:
                conf_analysis = st.sidebar.radio('Analyze confusion matrix:', ['Yes', 'No'])
                accuracy_analysis = st.sidebar.radio('Compare accuracy:', ['Yes', 'No'])
                precision_analysis = st.sidebar.radio('Compare precision:', ['Yes', 'No'])
                recall_analysis = st.sidebar.radio('Compare recall:', ['Yes', 'No'])
        
        elif i == 5:
            st.header('Model interpretation')
            st.subheader('Type of analysis')
            