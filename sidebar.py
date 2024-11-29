import streamlit as st
import pandas as pd
import time

def select_choice(var_name, options, intro_text='Select:'):
    if var_name not in st.session_state:
        st.session_state[var_name] = options[0]
    if st.session_state[var_name] not in options:
        st.session_state[var_name] = options[0]

    var = st.selectbox(
            intro_text,
            options,
            index = options.index(st.session_state[var_name]),
            key = var_name
        )

def radio_choice(var_name, options, intro_text='Choose:'):
    if var_name not in st.session_state:
        st.session_state[var_name] = options[0]
    if st.session_state[var_name] not in options:
        st.session_state[var_name] = options[0]

    var = st.sidebar.radio(
            'Choose problem type:', 
            options,
            index = options.index(st.session_state[var_name]),
            key = var_name
        )

def multiselect_choice(var_name, options, intro_text='Select/Unselect:', default_option='All'):
    if var_name not in st.session_state:
        if default_option == 'All':
            st.session_state[var_name] = options
        else:
            st.session_state[var_name] = []

    if st.session_state[var_name] not in options:
        if default_option == 'All':
            st.session_state[var_name] = options
        else:
            st.session_state[var_name] = [] 

    var = st.multiselect(
            'Select the input variables you want to use:', 
            options,
            default = st.session_state[var_name],
            key = var_name
        )


def sidebar_config(i):
    with st.sidebar:
        if i == 0:
            
            # Load data
            st.header('Input data')

            select_choice('col_sep', [',',';'] , 'What is the column separator of your file:')
            select_choice('dec_id', ['.',','] , 'What is the decimal point character:')

            uploaded_file = st.file_uploader("Upload the training set", type=["csv"])
            if uploaded_file is not None:
                st.session_state.uploaded  = True
                st.session_state.df_original = pd.read_csv(uploaded_file, sep=st.session_state.col_sep, index_col=False, decimal=st.session_state.dec_id)

        elif i == 1:

            # Select variables to analyze in detail
            st.header('Variable selection')
            select_choice('var_1', list(set(st.session_state.df_original.columns)) , 'Select a variable to analyze in detail:')
            select_choice('var_2', list(set(st.session_state.df_original.columns) - {st.session_state.var_1}) , 'Select a second variable to analyze in detail:')

        elif i == 2:

            st.header('Data preparation')
            st.write('Categorical data')
            select_choice(
                'categorical_treat', 
                ['Remove columns', 'Label encoding', 'One-hot encoding'], 
                'How to treat categorical data:'
            )

            st.write('Missing values treatment')
            select_choice(
                'missing_treat', 
                ['Remove observation', 'Imputation: mean', 'Imputation: median'], 
                'How to treat missing values:'
            )

            #Add option of how to find outliers
            st.write('Outlier treatment')
            select_choice(
                'outlier_treat', 
                ['Keep as-is', 'Remove observation', 'Imputation: mean', 'Imputation: median'], 
                'How to treat outlier values:'
            )

        elif i == 3:

            st.header('Problem Type')
            radio_choice(
                'problem_type', 
                ['Regression', 'Classification'], 
                'Choose problem type:'
            )

            st.header('Variable selection')
            select_choice(
                'to_predict', 
                list(set(st.session_state.df_treated.columns)), 
                'Select the variable you want to predict:'
            )
            multiselect_choice(
                'input_variables',
                list(set(st.session_state.df_treated.columns) - {to_predict}),
                'Select the input variables you want to use:',
                'All'
            )

            st.header('Training Parameters')
            parameter_split_size = st.slider(
                'Data split ratio (% for Training Set)', 
                min_value=10, 
                max_value=90, 
                value=80 if 'parameter_split_size' not in st.session_state else st.session_state.parameter_split_size, 
                step=5)
            st.session_state.parameter_split_size = parameter_split_size

            st.header('Model Parameters')
            st.subheader('Machine learning model')
            # Model Selection - Radio Button
            if problem_type == 'Regression':
                # Define the valid models for regression
                available_models = ['Linear regression', 'Random forest', 'Gradient boosting machines']
                
                # If a model was previously selected and it's not valid for regression, set to a default
                if 'model_to_use' in st.session_state:
                    if st.session_state.model_to_use not in available_models:
                        st.session_state.model_to_use = 'Linear regression'  # Default model for regression
                else:
                    st.session_state.model_to_use = 'Linear regression'  # Default model for first-time selection

            else:
                # Define the valid models for classification
                available_models = ['Logistic regression', 'Random forest', 'Gradient boosting machines']
                
                # If a model was previously selected and it's not valid for classification, set to a default
                if 'model_to_use' in st.session_state:
                    if st.session_state.model_to_use not in available_models:
                        st.session_state.model_to_use = 'Logistic regression'  # Default model for classification
                else:
                    st.session_state.model_to_use = 'Logistic regression'  # Default model for first-time selection

            # Display the model selection radio button
            model_to_use = st.sidebar.radio(
                'Choose model:', 
                available_models, 
                index=available_models.index(st.session_state.model_to_use)
            )
            st.session_state.model_to_use = model_to_use  # Save to session state

            st.subheader('Learning Parameters')
            #if model_to_use == 'Linear regression':
                # No additional parameters
            
            if model_to_use == 'Logistic regression':
                st.session_state.parameter_penalty = st.sidebar.radio(
                    'Penalty type (penalty)', ['l2', 'none'])
                st.session_state.parameter_c_value = st.sidebar.slider(
                    'Regularization strength (C)', 0.01, 10.0, 1.0, 0.01)
                st.session_state.parameter_solver = st.sidebar.radio(
                    'Solver', ['lbfgs', 'saga', 'liblinear'])
            
            if st.session_state.model_to_use in ['Random forest', 'Gradient boosting machines']:
                st.session_state.parameter_n_estimators = st.sidebar.slider(
                    'Number of estimators (n_estimators)', 5, 500, 100, 5)
                
                if st.session_state.model_to_use == 'Gradient boosting machines':
                    st.session_state.parameter_learning_rate = st.sidebar.slider(
                        'Learning rate', 0.01, 1.0, 0.1, 0.01)
                
                st.session_state.parameter_max_depth = st.sidebar.slider(
                    'Maximum depth of trees (max_depth)', 1, 100, 10, 1)
                
                st.session_state.parameter_min_samples_split = st.sidebar.slider(
                    'Minimum samples to split a node (min_samples_split)', 2, 20, 2, 1)
                
                st.session_state.parameter_min_samples_leaf = st.sidebar.slider(
                    'Minimum samples in leaf node (min_samples_leaf)', 1, 20, 1, 1)
                
                if st.session_state.problem_type == 'Regression':
                    if st.session_state.model_to_use == 'Random forest':
                        st.session_state.parameter_criterion = st.sidebar.radio(
                            'Performance measure (criterion)', 
                            ['squared_error', 'absolute_error', 'poisson', 'friedman_mse']
                        )
                    elif st.session_state.model_to_use == 'Gradient boosting machines':
                        st.session_state.parameter_criterion = st.sidebar.radio(
                            'Loss function (loss)', 
                            ['squared_error', 'absolute_error', 'huber', 'quantile']
                        )

                elif st.session_state.problem_type == 'Classification':
                    if st.session_state.model_to_use == 'Random forest':
                        st.session_state.parameter_criterion = st.sidebar.radio(
                            'Performance measure (criterion)', 
                            ['gini', 'entropy', 'log_loss']
                        )
                    elif st.session_state.model_to_use == 'Gradient boosting machines':
                        st.session_state.parameter_criterion = st.sidebar.radio(
                            'Loss function (loss)', 
                            ['log_loss', 'exponential']
                        )
                    st.session_state.balance_strat = st.sidebar.radio(
                            'Balancing strategy', 
                            ['None', 'Balance']
                        )

                #parameter_bootstrap = st.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
                #parameter_oob_score = st.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])

            st.header('Other Parameters')
            st.session_state.parameter_random_state = st.slider('Seed number (random_state)', 0, 1000, 42, 1)

        elif i == 4:

            st.header('Result Parameters')
            st.subheader('Metric analysis')
            if st.session_state.problem_type == 'Regression':
                st.session_state.mae_analysis = st.sidebar.radio('Compare Mean Absolute Error (MAE):', ['Yes', 'No'])
                st.session_state.mse_analysis = st.sidebar.radio('Compare Mean Squared Error (MSE):', ['Yes', 'No'])
                st.session_state.rmse_analysis = st.sidebar.radio('Compare Root Mean Squared Error (RMSE):', ['Yes', 'No'])
                st.session_state.r2_analysis = st.sidebar.radio('Compare R² Score:', ['Yes', 'No'])
                st.session_state.evs_analysis = st.sidebar.radio('Compare Explained Variance Score:', ['Yes', 'No'])

            else:
                st.session_state.conf_analysis = st.sidebar.radio('Analyze Confusion Matrix:', ['Yes', 'No'])
                st.session_state.accuracy_analysis = st.sidebar.radio('Compare Accuracy:', ['Yes', 'No'])
                st.session_state.precision_analysis = st.sidebar.radio('Compare Precision:', ['Yes', 'No'])
                st.session_state.recall_analysis = st.sidebar.radio('Compare Recall:', ['Yes', 'No'])
                st.session_state.f1_analysis = st.sidebar.radio('Compare F1 Score:', ['Yes', 'No'])
                st.session_state.auc_analysis = st.sidebar.radio('Analyze ROC AUC:', ['Yes', 'No'])
                st.session_state.logloss_analysis = st.sidebar.radio('Compare Log Loss:', ['Yes', 'No'])
        
        elif i == 5:
            st.header('Model interpretation')
            st.subheader('Analysis parameters')
            if st.session_state.model_to_use in ['Random forest', 'Gradient boosting machines']:
                st.session_state.var_analysis = st.selectbox(
                    'Select a variable to analyze in detail:',
                    st.session_state.x_train.columns
                )
                
        elif i == 6:
            st.header('Prediction data')

            col_sep_out = st.selectbox(
                'What is the column separator of your file:',
                [',',';']
            )
            dec_id_out = st.selectbox(
                'What is the decimal point character:',
                ['.',',']
            )

            uploaded_file_output = st.file_uploader("Upload the training set", type=["csv"])
            if uploaded_file_output is not None:
                st.session_state.predict_output  = True
                st.session_state.df_to_predict = pd.read_csv(uploaded_file_output, sep=col_sep_out, index_col=False, decimal=dec_id_out)

            st.session_state.download_everything = st.selectbox(
                'Do you want to download the training and test set:',
                ['Yes','No'],
                index=1
            )