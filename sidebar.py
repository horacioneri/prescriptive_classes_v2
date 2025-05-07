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

def radio_choice(var_name, options, intro_text='Choose:', default_option='first'):
    if var_name not in st.session_state:
        if default_option == 'first':
            st.session_state[var_name] = options[0]
        else:
            st.session_state[var_name] = default_option
    if st.session_state[var_name] not in options:
        if default_option == 'first':
            st.session_state[var_name] = options[0]
        else:
            st.session_state[var_name] = default_option

    var = st.sidebar.radio(
            intro_text, 
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
    if not set(st.session_state[var_name]).issubset(set(options)):
        if default_option == 'All':
            st.session_state[var_name] = options
        else:
            st.session_state[var_name] = []

    var = st.multiselect(
            intro_text, 
            options,
            default = st.session_state[var_name],
            key = var_name
        )

def slider_choice(var_name, slider_params, intro_text='Choose the value:'):
    if var_name not in st.session_state:
        st.session_state[var_name] = slider_params[2]
    var = st.sidebar.slider(
                        intro_text, 
                        slider_params[0], 
                        slider_params[1], 
                        st.session_state[var_name], 
                        slider_params[3],
                        key = var_name
                    )


def sidebar_config(i):
    with st.sidebar:
        if i == 1:
            
            # Load data
            st.header('Input data')

            select_choice('col_sep', [',',';'] , 'What is the column separator of your file:')
            select_choice('dec_id', ['.',','] , 'What is the decimal point character:')

            uploaded_file = st.file_uploader("Upload the training set", type=["csv"])
            if uploaded_file is not None:
                st.session_state.uploaded  = True
                st.session_state.df_original = pd.read_csv(uploaded_file, sep=st.session_state.col_sep, index_col=False, decimal=st.session_state.dec_id)

        elif i == 2:

            # Select variables to analyze in detail
            st.header('Variable selection')
            select_choice('var_1', list(set(st.session_state.df_original.columns)) , 'Select a variable to analyze in detail:')
            select_choice('var_2', list(set(st.session_state.df_original.columns) - {st.session_state.var_1}) , 'Select a second variable to analyze in detail:')

        elif i == 3:

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

        elif i == 4:

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
                list(set(st.session_state.df_treated.columns) - {st.session_state.to_predict}),
                'Select the input variables you want to use:',
                'All'
            )

            st.header('Training Parameters')
            slider_choice('parameter_split_size', [10,90,80,5],'Data split ratio (% for Training Set)')

            st.header('Model Parameters')
            st.subheader('Machine learning model')
            if st.session_state.problem_type == 'Regression':
                # Define the valid models for regression
                available_models = ['Linear regression', 'Random forest', 'Gradient boosting machines']
            else:
                available_models = ['Logistic regression', 'Random forest', 'Gradient boosting machines']

            radio_choice(
                'model_to_use', 
                available_models, 
                'Choose predictive model:'
            )

            st.subheader('Learning Parameters')
            #if model_to_use == 'Linear regression':
                # No additional parameters
            
            if st.session_state.model_to_use == 'Logistic regression':
                radio_choice('parameter_penalty',['l2', 'none'],'Penalty type (penalty)')
                slider_choice('parameter_c_value', [0.01, 10.0, 1.0, 0.01], 'Regularization strength (C)')
                radio_choice('parameter_solver',['lbfgs', 'saga', 'liblinear'],'Solver')
            
            if st.session_state.model_to_use in ['Random forest', 'Gradient boosting machines']:
                slider_choice('parameter_n_estimators', [5, 500, 100, 5], 'Number of estimators (n_estimators)')

                if st.session_state.model_to_use == 'Gradient boosting machines':
                    slider_choice('parameter_learning_rate', [0.01, 1.0, 0.1, 0.01], 'Learning rate')
                
                slider_choice('parameter_max_depth', [1, 100, 10, 1], 'Maximum depth of trees (max_depth)')
                slider_choice('parameter_min_samples_split', [2, 20, 2, 1], 'Minimum samples to split a node (min_samples_split)')
                slider_choice('parameter_min_samples_leaf', [1, 20, 1, 1], 'Minimum samples in leaf node (min_samples_leaf)')
                
                if st.session_state.problem_type == 'Regression':
                    if st.session_state.model_to_use == 'Random forest':
                        radio_choice('parameter_criterion', ['squared_error', 'absolute_error', 'poisson', 'friedman_mse'], 'Performance measure (criterion)')

                    elif st.session_state.model_to_use == 'Gradient boosting machines':
                        radio_choice('parameter_criterion', ['squared_error', 'absolute_error', 'huber', 'quantile'], 'Loss function (loss)')

                elif st.session_state.problem_type == 'Classification':
                    if st.session_state.model_to_use == 'Random forest':
                        radio_choice('parameter_criterion', ['gini', 'entropy', 'log_loss'], 'Performance measure (criterion)')
                        radio_choice('balance_strat', ['None', 'Balance'], 'Balancing strategy')

                    elif st.session_state.model_to_use == 'Gradient boosting machines':
                        radio_choice('parameter_criterion', ['log_loss', 'exponential'], 'Loss function (loss)')

                #parameter_bootstrap = st.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
                #parameter_oob_score = st.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])

            st.header('Other Parameters')
            slider_choice('parameter_random_state', [0, 1000, 42, 1], 'Seed number (random_state)')

        elif i == 5:

            st.header('Result Parameters')
            st.subheader('Metric analysis')
            if st.session_state.problem_type == 'Regression':
                radio_choice('mae_analysis', ['Yes', 'No'], 'Compare Mean Absolute Error (MAE):')
                radio_choice('mse_analysis', ['Yes', 'No'], 'Compare Mean Squared Error (MSE):')
                radio_choice('rmse_analysis', ['Yes', 'No'], 'Compare Root Mean Squared Error (RMSE):')
                radio_choice('r2_analysis', ['Yes', 'No'], 'Compare R² Score:')
                radio_choice('evs_analysis', ['Yes', 'No'], 'Compare Explained Variance Score:', 'No')

            else:
                radio_choice('conf_analysis', ['Yes', 'No'], 'Analyze Confusion Matrix:')
                radio_choice('accuracy_analysis', ['Yes', 'No'], 'Compare Accuracy:')
                radio_choice('precision_analysis', ['Yes', 'No'], 'Compare Precision:')
                radio_choice('recall_analysis', ['Yes', 'No'], 'Compare Recall:')
                radio_choice('f1_analysis', ['Yes', 'No'], 'Compare F1 Score:')
                radio_choice('auc_analysis', ['Yes', 'No'], 'Analyze ROC AUC:', 'No')
                radio_choice('logloss_analysis', ['Yes', 'No'], 'Compare Log Loss:', 'No')
        
        elif i == 6:
            st.header('Model interpretation')
            st.subheader('Analysis parameters')
            if st.session_state.model_to_use in ['Random forest', 'Gradient boosting machines']:
                radio_choice('traditional_imp', ['Yes', 'No'], 'Analyze traditional feature importance:', 'Yes')
                radio_choice('permutation_imp', ['Yes', 'No'], 'Analyze permutation feature importance:', 'No')
                radio_choice('shap_analysis', ['Yes', 'No'], 'Analyze shap values:', 'No')
                radio_choice('partial_dep_plot', ['Yes', 'No'], 'Analyze partial dependence plot:', 'No')
                if st.session_state.partial_dep_plot == 'Yes':
                    select_choice('var_analysis', list(set(st.session_state.x_train.columns)), 'Select a variable to analyze in detail:')
                
        elif i == 7:
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