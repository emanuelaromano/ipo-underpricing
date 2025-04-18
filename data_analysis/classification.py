def evaluate_best_model_on_test_set(model, df):
    """
    Evaluate the best model on the test set and save the results.
    
    Args:
        model: The trained model to evaluate
        df: The test dataset DataFrame
    """
    # Load test data
    df_test = pd.read_csv('./output_csv/Final_Output_Class_test.csv')
    X_test, y_test = split_data(df_test)
    
    # Preprocess the test set
    # Fill missing values with the mode for categorical columns and mean for numerical columns
    for col in X_test.columns:
        if X_test[col].dtype == 'object':
            mode = X_test[col].mode()[0]
            X_test[col] = X_test[col].fillna(mode)
        else:
            mean = X_test[col].mean()
            X_test[col] = X_test[col].fillna(mean)
    
    # Apply the same preprocessing as in the encoding function
    oe_columns = ['Trade Month', 'Trade Day', 'Trade Year']
    ohe_columns = ['Industry Sector', 'Industry Group', 'Industry Subgroup']
    ss_columns = [col for col in X_test.select_dtypes(exclude=['object']).columns if col not in oe_columns]
    
    # Get the maximum value from the training data for unknown values in OrdinalEncoder
    max_value = df[oe_columns].max().max()
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=int(max_value))
    ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    ss = StandardScaler()
    
    preprocess = ColumnTransformer(transformers=[
        ('ohe', ohe, ohe_columns),
        ('ss', ss, ss_columns),
        ('oe', oe, oe_columns)
    ], remainder='passthrough')
    
    # Fit on training data and transform test data
    preprocess.fit(df.drop(columns=['Underpriced']))
    transformed_test = preprocess.transform(X_test)
    features = preprocess.get_feature_names_out()
    X_test = pd.DataFrame(transformed_test, columns=features)
    
    # Clean column names
    X_test = clean_column_names(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    test_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0)
    }
    
    metrics_df = pd.DataFrame([test_metrics])
    metrics_df.to_csv('./output_csv/best_model_test_results.csv', index=False)
    
    print("\nTest Set Performance:")
    display(metrics_df)
    
    return test_metrics 