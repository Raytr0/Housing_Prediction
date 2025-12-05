import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import os
import glob
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_data():
    df = None
    try:
        path = kagglehub.dataset_download("yasserh/housing-prices-dataset")
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        filepath = csv_files[0]
        df = pd.read_csv(filepath)
        return df

    except Exception as e:
        print(f"Automatic download failed: {e}")

    local_filepath = 'Housing.csv'
    if os.path.exists(local_filepath):
        df = pd.read_csv(local_filepath)
        return df

    raise FileNotFoundError(
        "Could not find 'Housing.csv'"
    )


def build_model():
    # Define features
    categorical_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating',
                            'airconditioning', 'prefarea', 'furnishingstatus']
    numerical_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']

    # Preprocessing
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Define the model (XGBoost)
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        #('regressor', XGBRegressor(n_estimators=500, learning_rate=0.05, n_jobs=-1, random_state=42))
        ('regressor', XGBRegressor(
            n_estimators=2000,  # Increased from 500: Gives the model more chances to correct errors.
            learning_rate=0.01,  # Decreased from 0.05: Makes the model learn slower but more precisely.
            max_depth=6,  # Limits how complex the trees can get (prevents memorizing the data).
            subsample=0.8,  # Uses only 80% of rows per tree to prevent overfitting.
            colsample_bytree=0.8,  # Uses only 80% of features per tree to add variety.
            n_jobs=-1,
            random_state=42
        ))
    ])

    return model


def visualize_model_performance(model, X_test, y_test, predictions):
    sns.set(style="whitegrid")

    # Chart 1: Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.6, color='b')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Prices (XGBoost)')
    plt.tight_layout()
    plt.show()

    # Chart 2: Feature Importance
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    importances = model.named_steps['regressor'].feature_importances_

    feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_df = feat_df.sort_values(by='Importance', ascending=False).head(10)  # Top 10 only

    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_df, palette='viridis')
    plt.title('Top 10 Feature Importances (XGBoost)')
    plt.tight_layout()
    plt.show()

    # Chart 3: Residuals Distribution
    residuals = y_test - predictions
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color='purple')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Residuals (Error)')
    plt.title('Distribution of Residuals')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = load_data()

    # Separate Target (price) and Features (X)
    X = df.drop('price', axis=1)
    y = df['price']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and Train
    model = build_model()
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Model Performance:")
    print(f"Mean Absolute Error: ${mae:,.2f}")
    print(f"Accuracy (R² Score): {r2:.2%}")

    visualize_model_performance(model, X_test, y_test, predictions)

    try:
        user_input = {
            'area': [7000],
            'bedrooms': [3],
            'bathrooms': [2],
            'stories': [2],
            'mainroad': ['yes'],
            'guestroom': ['no'],
            'basement': ['no'],
            'hotwaterheating': ['no'],
            'airconditioning': ['yes'],
            'parking': [1],
            'prefarea': ['yes'],
            'furnishingstatus': ['semi-furnished']
        }

        input_df = pd.DataFrame(user_input)

        # Predict
        predicted_price = model.predict(input_df)[0]
        print(f"\nEstimated Price: ${predicted_price:,.2f}")

    except ValueError:
        print("\nInvalid input. Check input fields.")