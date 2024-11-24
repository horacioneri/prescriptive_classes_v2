import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, log_loss
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
import statsmodels.api as sm
import shap
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

def exploratory_data_analysis():
    df = st.session_state.df_original
    st.header('Single variable analysis', divider='rainbow')
    col = st.columns(2)
    for c in range(len(col)):
        if c == 0:
            var = st.session_state.var_1
        else:
            var = st.session_state.var_2

        with col[c]:
            st.subheader(var)
            var_data = df[var]
            if var_data.dtype in ['int64', 'float64']:
                st.write(var_data.describe())

                # Visualize the distribution (Histogram with Plotly)
                fig = px.histogram(var_data, nbins=20, title=f'Distribution of {var}')
                fig.update_layout(
                    xaxis_title=var,
                    yaxis_title='Frequency',
                    template="seaborn",  # Choose a template (e.g., "plotly_dark", "ggplot2", etc.)
                    showlegend=True,
                    legend = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)

                # Box plot to detect outliers
                fig = px.box(var_data, title=f'Box plot of {var}')
                fig.update_layout(
                    yaxis_title=var,
                    template="seaborn",  # Choose a template (e.g., "plotly_dark", "ggplot2", etc.)
                    showlegend=True,
                    legend = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.write(var_data.value_counts())

                # Bar plot for category distribution
                fig = px.bar(var_data.value_counts().reset_index(), x=var, y='count', 
                            title=f'Count plot of {var}', labels={var: var, 'count': 'Frequency'})
                fig.update_layout(
                    template="seaborn",  # Choose a template (e.g., "plotly_dark", "ggplot2", etc.)
                    showlegend=True,
                    legend = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)

                # Pie chart for proportions
                fig = px.pie(var_data, names=var_data.value_counts().index, 
                            title=f'Pie chart of {var}', 
                            hole=0.3)
                fig.update_traces(textinfo='percent+label')
                fig.update_layout(
                    template="seaborn",  # Choose a template (e.g., "plotly_dark", "ggplot2", etc.)
                    showlegend=True,
                    legend = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
        
    st.header('Two variable analysis', divider='rainbow')
    var_name1 = st.session_state.var_1
    var_name2 = st.session_state.var_2
    var_data1 = df[var_name1]
    var_data2 = df[var_name2]

    # Case 1: Both variables are numerical
    if var_data1.dtype in ['int64', 'float64'] and var_data2.dtype in ['int64', 'float64']:
        # Scatter plot to show relationship
        fig = px.scatter(df, x=var_name1, y=var_name2, trendline="ols", title=f"Scatter plot of {var_name1} vs {var_name2} with linear regression line")
        fig.update_layout(
                    xaxis_title=var_name1, 
                    yaxis_title=var_name2,
                    template="seaborn",  # Choose a template (e.g., "plotly_dark", "ggplot2", etc.)
                    showlegend=True,
                    legend = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
        st.plotly_chart(fig, use_container_width=True)

        # Correlation coefficient
        corr = var_data1.corr(var_data2)
        print(f"Correlation coefficient between {var_name1} and {var_name2}: {corr}")

    # Case 2: One variable is numerical and the other is categorical
    elif var_data1.dtype in ['int64', 'float64'] and var_data2.dtype in ['object', 'category']:
        # Box plot
        fig = px.box(df, x=var_name2, y=var_name1, title=f"Box plot of {var_name1} by {var_name2}")
        fig.update_layout(
                    xaxis_title=var_name1, 
                    yaxis_title=var_name2,
                    template="seaborn",  # Choose a template (e.g., "plotly_dark", "ggplot2", etc.)
                    showlegend=True,
                    legend = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
        st.plotly_chart(fig, use_container_width=True)
    
    # Case 3: Both variables are categorical
    elif var_data1.dtype in ['object', 'category'] and var_data2.dtype in ['object', 'category']:
        # Stacked bar plot
        contingency_table = pd.crosstab(df[var_name1], df[var_name2])
        fig = px.bar(contingency_table, barmode='stack', title=f"Stacked bar plot of {var_name1} and {var_name2}")
        fig.update_layout(
                    xaxis_title=var_name1, 
                    yaxis_title='Count',
                    template="seaborn",  # Choose a template (e.g., "plotly_dark", "ggplot2", etc.)
                    showlegend=True,
                    legend = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
        st.plotly_chart(fig, use_container_width=True)

        # Heatmap of the contingency table
        fig = go.Figure(data=go.Heatmap(z=contingency_table.values, x=contingency_table.columns, y=contingency_table.index,
                                    colorscale='Viridis'))
        fig.update_layout(
                    title=f"Heatmap of {var_name1} vs {var_name2}", 
                    xaxis_title=var_name2, 
                    yaxis_title=var_name1,
                    template="seaborn",  # Choose a template (e.g., "plotly_dark", "ggplot2", etc.)
                    showlegend=True,
                    legend = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
        st.plotly_chart(fig, use_container_width=True)
    
    # Case 4: Same as 2 but the other way around
    else:
        # Box plot
        fig = px.box(df, x=var_name1, y=var_name2, title=f"Box plot of {var_name2} by {var_name1}")
        fig.update_layout(
                    xaxis_title=var_name2, 
                    yaxis_title=var_name1,
                    template="seaborn",  # Choose a template (e.g., "plotly_dark", "ggplot2", etc.)
                    showlegend=True,
                    legend = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
        st.plotly_chart(fig, use_container_width=True)

    st.header('Correlation analysis', divider='rainbow')
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    
    # Calculate the correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Plot the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',  # Seaborn-like diverging colorscale
        colorbar=dict(title="Correlation Coefficient", ticksuffix="", outlinewidth=0),
        zmin=-1, zmax=1,
        hovertemplate="X: %{x}<br>Y: %{y}<br>Correlation: %{z:.2f}<extra></extra>"
    ))

    # Add correlation coefficients as text annotations
    fig.update_traces(
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",  # Format annotations
        textfont=dict(size=10)#,  # Smaller text to avoid clutter
        #hoverinfo='text'
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text="Correlation Matrix of Numeric Variables",
            x=0,  # Left align the title
            xanchor='left'
        ),
        xaxis=dict(
            tickangle=45,
            showgrid=False,  # No gridlines
            zeroline=False,
            showticklabels=True
        ),
        yaxis=dict(
            tickangle=0,
            showgrid=False,  # No gridlines
            zeroline=False,
            showticklabels=True
        ),
        font=dict(
            family="Arial",  # Similar to Seaborn's default
            size=12
        ),
        template="seaborn"  # Use the built-in "seaborn" template
    )

    # Show the plot
    st.plotly_chart(fig, use_container_width=True)

def data_preparation():
    st.header('Data preparation', divider='rainbow')
    st.subheader('Treating categorical columns')
    df = st.session_state.df_original

    # Identify categorical columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    if st.session_state.categorical_treat == 'Remove columns':
        df = df.drop(columns=categorical_columns)
    elif st.session_state.categorical_treat == 'Label encoding':
        # Apply Label Encoding to each categorical column
        for col in categorical_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    elif st.session_state.categorical_treat == 'One-hot encoding':
        # Apply One-hot encoding to each categorical column
        # Initialize the OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False, drop=None)

        # Apply one-hot encoding
        encoded_data = encoder.fit_transform(df[categorical_columns])

        # Create a DataFrame for the encoded columns
        encoded_df = pd.DataFrame(
            encoded_data,
            columns=encoder.get_feature_names_out(categorical_columns)
        )

        # Combine with the original DataFrame (excluding the original categorical columns)
        df = pd.concat([df.drop(columns=categorical_columns), encoded_df], axis=1)
    
    st.write(f"After applying the method '{st.session_state.categorical_treat}' to the categorical columns, your dataset looks like:")
    st.dataframe(df, height = 300)

    st.subheader('Treating missing values')
    if st.session_state.missing_treat == 'Remove observation':
        df = df.dropna()
    elif st.session_state.missing_treat == 'Imputation: mean':
        df = df.fillna(df.mean())
    elif st.session_state.missing_treat == 'Imputation: median':
        df = df.fillna(df.median())

    st.write(f"After applying the method '{st.session_state.missing_treat}' to the missing values, your dataset looks like:")
    st.dataframe(df, height = 300)

    st.subheader('Treating outlier values')
    if st.session_state.outlier_treat != 'Keep as-is':
        # Find binary variables to exclude from outlier analysis
        bin_vars = [c for c in df.columns if set(df[c].unique()) == {0, 1}]

        # Calculate the first (25th percentile) and third (75th percentile) quartiles
        Q1 = df.select_dtypes(include=['float64', 'int64']).quantile(0.25)
        Q3 = df.select_dtypes(include=['float64', 'int64']).quantile(0.75)

        # Calculate IQR (Interquartile Range)
        IQR = Q3 - Q1

        # Determine the lower and upper bounds for each column
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Initialize a mask that will indicate rows to keep (True means keep the row)
        keep_rows_mask = pd.Series(True, index=df.index)

        # Iterate through each numerical column (excluding one-hot encoded columns) to identify and remove outliers
        for c in df.select_dtypes(include=['float64', 'int64']).columns:
            if c not in bin_vars:  # Skip one-hot encoded columns
                # Identify outliers (values outside the bounds)
                outliers = (df[c] < lower_bound[c]) | (df[c] > upper_bound[c])

                if st.session_state.outlier_treat == 'Remove observation':
                    # Mark rows with outliers as False in the mask
                    keep_rows_mask &= ~outliers  # Only keep rows that don't have outliers

                    # Return the DataFrame with rows removed that had outliers in any numerical column
                    df = df[keep_rows_mask]
                
                elif st.session_state.outlier_treat == 'Imputation: mean':
                    # Replace outliers with the mean of the column
                    df[c] = df[c].where(~outliers, df[c].mean())

                elif st.session_state.outlier_treat == 'Imputation: median':
                    # Replace outliers with the mean of the column
                    df[c] = df[c].where(~outliers, df[c].median())
        
    st.write(f"After applying the method '{st.session_state.outlier_treat}' to the outlier values, your dataset looks like:")
    st.dataframe(df, height = 300)
    st.session_state.treated = True
    st.session_state.df_treated = df

def model_training():
    df = st.session_state.df_treated
    y = df[st.session_state.to_predict]
    x = df[st.session_state.input_variables]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(100-st.session_state.parameter_split_size)/100, random_state=st.session_state.parameter_random_state)

    # Summarize the dataset
    st.header('Dataset split', divider='rainbow')
    rows, cols = x_train.shape
    st.write(f"Your training set has {rows} observations and {cols} input variables.")
    rows, cols = x_test.shape
    st.write(f"Your test set has {rows} observations and {cols} input variables.")

    # Start training
    st.header('Model training', divider='rainbow')
    if st.session_state.problem_type == 'Regression':
        if st.session_state.model_to_use == 'Linear regression':
            # Linear Regression does not require additional hyperparameters in most cases
            ml_mod = LinearRegression()

        elif st.session_state.model_to_use == 'Random forest':
            ml_mod = RandomForestRegressor(
                n_estimators=st.session_state.parameter_n_estimators,
                random_state=st.session_state.parameter_random_state,
                criterion=st.session_state.parameter_criterion,
                max_depth=st.session_state.parameter_max_depth,
                min_samples_split=st.session_state.parameter_min_samples_split,
                min_samples_leaf=st.session_state.parameter_min_samples_leaf
            )

        elif st.session_state.model_to_use == 'Gradient boosting machines':
            st.write('preparing GradientBoostingRegressor')
            ml_mod = GradientBoostingRegressor(
                loss=st.session_state.parameter_criterion,
                n_estimators=st.session_state.parameter_n_estimators,
                learning_rate=st.session_state.parameter_learning_rate,
                max_depth=st.session_state.parameter_max_depth,
                min_samples_split=st.session_state.parameter_min_samples_split,
                min_samples_leaf=st.session_state.parameter_min_samples_leaf,
                random_state=st.session_state.parameter_random_state
            )

    elif st.session_state.problem_type == 'Classification':
        if st.session_state.model_to_use == 'Logistic regression':
            ml_mod = LogisticRegression(
                penalty=st.session_state.parameter_penalty,
                C=st.session_state.parameter_c_value,
                solver=st.session_state.parameter_solver,
                random_state=st.session_state.parameter_random_state
            )

        elif st.session_state.model_to_use == 'Random forest':
            if st.session_state.balance_strat == 'Balanced':
                class_weights = 'balanced'
            else:
                class_weights = None
            
            ml_mod = RandomForestClassifier(
                n_estimators=st.session_state.parameter_n_estimators,
                random_state=st.session_state.parameter_random_state,
                criterion=st.session_state.parameter_criterion,
                max_depth=st.session_state.parameter_max_depth,
                min_samples_split=st.session_state.parameter_min_samples_split,
                min_samples_leaf=st.session_state.parameter_min_samples_leaf,
                class_weight=class_weights
            )

        elif st.session_state.model_to_use == 'Gradient boosting machines':
            if st.session_state.balance_strat == 'Balanced':
                class_weights = compute_sample_weight(
                    class_weight='balanced', y=y_train)
            else:
                class_weights = None

            ml_mod = GradientBoostingClassifier(
                loss=st.session_state.parameter_criterion,
                n_estimators=st.session_state.parameter_n_estimators,
                learning_rate=st.session_state.parameter_learning_rate,
                max_depth=st.session_state.parameter_max_depth,
                min_samples_split=st.session_state.parameter_min_samples_split,
                min_samples_leaf=st.session_state.parameter_min_samples_leaf,
                random_state=st.session_state.parameter_random_state
            )

    if st.session_state.problem_type == 'Classification' and st.session_state.model_to_use == 'Gradient boosting machines':
        ml_mod.fit(x_train, y_train, sample_weight=class_weights)
    else:
        ml_mod.fit(x_train, y_train)

    y_train_pred = ml_mod.predict(x_train)
    y_test_pred = ml_mod.predict(x_test)

    st.write('Your model has finished training, see below the predictions for the training and tests:')
    col = st.columns(2)
    with col[0]:
        st.subheader('Train set')
        df_y_train_pred = pd.DataFrame(y_train_pred, columns=['pred'])
        x_train = x_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        df_y_train_pred = df_y_train_pred.reset_index(drop=True)
        st.dataframe(pd.concat([x_train, y_train, df_y_train_pred], axis=1), height = 300)

    with col[1]:
        st.subheader('Test set')
        df_y_test_pred = pd.DataFrame(y_test_pred, columns=['pred'])
        x_test = x_test.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        df_y_test_pred = df_y_test_pred.reset_index(drop=True)
        st.dataframe(pd.concat([x_test, y_test, df_y_test_pred], axis=1), height = 300)

    st.session_state.trained = True
    st.session_state.x_train = x_train
    st.session_state.x_test = x_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    st.session_state.y_train_pred = df_y_train_pred
    st.session_state.y_test_pred = df_y_test_pred
    st.session_state.ml_mod = ml_mod

def result_analysis():
    st.header('Analysis of result metrics', divider='rainbow')
    
    col = st.columns(2)
    for c in range(len(col)):
        with col[c]:
            if c == 0:
                st.subheader('Train set')
                y = st.session_state.y_train
                y_pred = st.session_state.y_train_pred
            else:
                st.subheader('Test set')
                y = st.session_state.y_test
                y_pred = st.session_state.y_test_pred
            
            if st.session_state.problem_type == 'Regression':
                if st.session_state.mae_analysis:
                    mae = mean_absolute_error(y, y_pred)
                    st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
                if st.session_state.mse_analysis:
                    mse = mean_squared_error(y, y_pred)
                    st.write(f"Mean Squared Error (MSE): {mse:.4f}")
                if st.session_state.rmse_analysis:
                    rmse = np.sqrt(mean_squared_error(y, y_pred))
                    st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
                if st.session_state.r2_analysis:
                    r2 = r2_score(y, y_pred)
                    st.write(f"R² Score: {r2:.4f}")
                if st.session_state.evs_analysis:
                    evs = explained_variance_score(y, y_pred)
                    st.write(f"Explained Variance Score: {evs:.4f}")
            else:
                if st.session_state.conf_analysis:
                    # Calculate the confusion matrix
                    conf_matrix = confusion_matrix(y, y_pred)
                    labels = [f"Class {i}" for i in range(len(conf_matrix))]  # Modify based on your class labels

                    # Create a heatmap with Plotly
                    fig = go.Figure(
                        data=go.Heatmap(
                            z=conf_matrix,
                            x=labels,
                            y=labels,
                            colorscale="Blues",
                            colorbar=dict(title="Count"),
                            hoverongaps=False,
                            text=conf_matrix,  # Annotate the heatmap with counts
                            texttemplate="%{text}",  # Display values on the heatmap
                            textfont=dict(size=12)  # Text font size for annotations
                        )
                    )

                    # Update layout for better readability
                    fig.update_layout(
                        title="Confusion Matrix",
                        xaxis_title="Predicted Label",
                        yaxis_title="True Label",
                        xaxis=dict(tickmode='linear'),
                        yaxis=dict(tickmode='linear'),
                        template="seaborn",  # Choose a template (e.g., "plotly_dark", "ggplot2", etc.)
                        showlegend=True,
                        legend = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )

                    # Display the confusion matrix
                    st.plotly_chart(fig, use_container_width=True)

                if st.session_state.accuracy_analysis:
                    acc = accuracy_score(y, y_pred)
                    st.write(f"Accuracy: {acc:.4f}")
                if st.session_state.precision_analysis:
                    prec = precision_score(y, y_pred, average='weighted')
                    st.write(f"Precision: {prec:.4f}")
                if st.session_state.recall_analysis:
                    rec = recall_score(y, y_pred, average='weighted')
                    st.write(f"Recall: {rec:.4f}")
                if st.session_state.f1_analysis:
                    f1 = f1_score(y, y_pred, average='weighted')
                    st.write(f"F1 Score: {f1:.4f}")
                if st.session_state.auc_analysis:
                    roc_auc = roc_auc_score(y, y_pred, multi_class='ovr')  # Adjust for multi-class
                    st.write(f"ROC AUC Score: {roc_auc:.4f}")

                    fpr, tpr, _ = roc_curve(y, y_pred)
                    roc_auc = auc(fpr, tpr)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"ROC Curve (AUC = {roc_auc:.2f})"))
                    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random Guess'))
                    fig.update_layout(
                        title="ROC Curve", 
                        xaxis_title="False Positive Rate", 
                        yaxis_title="True Positive Rate",
                        template="seaborn",  # Choose a template (e.g., "plotly_dark", "ggplot2", etc.)
                        showlegend=True,
                        legend = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                if st.session_state.logloss_analysis:
                    logloss = log_loss(y, y_pred)
                    st.write(f"Log Loss: {logloss:.4f}")

def model_interpretation():
    
    x_train = st.session_state.x_train
    y_train = st.session_state.y_train
    x_test = st.session_state.x_test
    y_test = st.session_state.y_test
    ml_mod = st.session_state.ml_mod
    
    if st.session_state.model_to_use == 'Linear regression':
        st.header("Linear regression analysis", divider='rainbow')

        # Fit statsmodels for additional stats (replace with actual data)
        x_train_sm = sm.add_constant(x_train)  # Add intercept term
        ols_model = sm.OLS(y_train, x_train_sm).fit()

        # Intercept and coefficients
        st.subheader("Coefficients")
        coeffs_df = pd.DataFrame({
            "Feature": ["Intercept"] + list(x_train.columns),
            "Coefficient": [ols_model.params[0]] + list(ols_model.params[1:]),
            "P-Value": [ols_model.pvalues[0]] + list(ols_model.pvalues[1:]),
            "95% CI Lower": ols_model.conf_int().iloc[:, 0],
            "95% CI Upper": ols_model.conf_int().iloc[:, 1]
        })
        coeffs_df = coeffs_df.sort_values("Coefficient", key=abs, ascending=False)
        st.dataframe(coeffs_df)

        # R² and Adjusted R²
        st.subheader("R² Metrics")
        st.write(f"R²: {ols_model.rsquared:.4f}")
        st.write(f"Adjusted R²: {ols_model.rsquared_adj:.4f}")

    elif st.session_state.model_to_use == 'Logistic regression':
        st.header("Logistic regression analysis", divider='rainbow')

        # Fit statsmodels for additional stats (replace with actual data)
        x_train_sm = sm.add_constant(x_train)  # Add intercept term
        logit_model = sm.Logit(y_train, x_train_sm).fit()

        # Coefficients and odds ratios
        st.subheader("Coefficients and Odds Ratios")
        coeffs_df = pd.DataFrame({
            "Feature": ["Intercept"] + list(x_train.columns),
            "Coefficient": [logit_model.params[0]] + list(logit_model.params[1:]),
            "Odds Ratio": np.exp([logit_model.params[0]] + list(logit_model.params[1:])),
            "P-Value": [logit_model.pvalues[0]] + list(logit_model.pvalues[1:]),
            "95% CI Lower": logit_model.conf_int().iloc[:, 0],
            "95% CI Upper": logit_model.conf_int().iloc[:, 1]
        })
        coeffs_df = coeffs_df.sort_values("Odds Ratio", key=abs, ascending=False)
        st.dataframe(coeffs_df)

    elif st.session_state.model_to_use in ['Random forest', 'Gradient boosting machines']:
        st.header("Machine learning model analysis", divider='rainbow')

        # Traditional feature importance
        st.subheader("Traditional Feature Importance")
        feature_importance = pd.Series(ml_mod.feature_importances_, index=x_train.columns)
        feature_importance = feature_importance.sort_values(ascending=False)
        st.bar_chart(feature_importance)

        # Permutation importance
        st.subheader("Permutation Importance")
        perm_importance = permutation_importance(ml_mod, x_test, y_test, n_repeats=10, random_state=st.session_state.parameter_random_state)
        perm_importance_df = pd.DataFrame({
            "Feature": x_train.columns,
            "Importance": perm_importance.importances_mean
        }).sort_values("Importance", ascending=False)
        st.dataframe(perm_importance_df)

        # SHAP values
        st.subheader("SHAP Values")
        explainer = shap.TreeExplainer(ml_mod)
        shap_values = explainer.shap_values(x_test)
        shap.summary_plot(shap_values, x_test, show=False)  # Suppress direct output
        st.pyplot(bbox_inches='tight')

        # Partial dependence plots
        st.subheader("Partial Dependence Plots")
        top_features = feature_importance.index[:2]  # Select top 2 features
        fig, ax = plt.subplots(figsize=(10, 6))
        PartialDependenceDisplay.from_estimator(ml_mod, x_test, top_features, ax=ax)
        st.pyplot(fig)
