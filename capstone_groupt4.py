import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def load_and_inspect_data(file_path):
    try:
        mlb_df = pd.read_csv(file_path)
        print("--- 1. Data Loading and Inspection ---")
        print("Data loaded successfully.")
        print(f"Shape of the dataset: {mlb_df.shape}")
        print("\nFirst 5 rows:")
        print(mlb_df.head())
        print("\nBasic Info:")
        mlb_df.info()
        print("-" * 50)
        return mlb_df
    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}")
        return None

def perform_linear_regression_analysis(mlb_df):
    print("\n--- 2. Building Linear Regression Model (Predicting ERA) ---")
    
    # --- 1. Data Preparation ---
    mlb_df_cleaned = mlb_df.copy()
    
    # --- Feature Engineering ---
    mlb_df_cleaned['SO9'] = mlb_df_cleaned.apply(lambda row: (row['SO'] / row['IP'] * 9) if row['IP'] > 0 else 0, axis=1)
    mlb_df_cleaned['HR9'] = mlb_df_cleaned.apply(lambda row: (row['HR'] / row['IP'] * 9) if row['IP'] > 0 else 0, axis=1)

    # --- features list ---
    reg_features = ['WAR', 'SO9', 'HR9', 'WHIP'] 
    reg_target = 'ERA'
    
    model_df = mlb_df_cleaned[reg_features + [reg_target]].copy()

    # Missing value -> median
    imputer = SimpleImputer(strategy='median')
    model_df[reg_features] = imputer.fit_transform(model_df[reg_features])
    
    # Drop missing targe value
    model_df.dropna(subset=[reg_target], inplace=True)

    X = model_df[reg_features]
    y = model_df[reg_target]

    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train set and test set
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # --- 2. Model Training ---
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    # --- 3. Evaluation ---
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"R-squared: {r2:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    
    coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
    
    print("\nImpact of Each Stat on ERA (Coefficients):")
    print(coefficients)
    
    # --- 4. Visualization ---
    plt.figure(figsize=(5, 4))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
    plt.title('Actual ERA vs. Predicted ERA (Linear Model)')
    plt.xlabel('Actual ERA')
    plt.ylabel('Predicted ERA')
    plt.show()