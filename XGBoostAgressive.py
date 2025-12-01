import pandas as pd
import numpy as np
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


def load_data():
    df = None
    try:
        path = kagglehub.dataset_download("yasserh/housing-prices-dataset")
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        filepath = csv_files[0]
        df = pd.read_csv(filepath)
    except Exception:
        local_filepath = 'Housing.csv'
        if os.path.exists(local_filepath):
            df = pd.read_csv(local_filepath)
        else:
            raise FileNotFoundError("Could not find dataset.")
    return df


# --- Feature Engineering Function ---
def enrich_data(df):
    """
    Creates new 'smart' features from existing columns.
    We apply this to BOTH the training data and the User Input.
    """
    df = df.copy()

    # Feature 1: Area per Room (Crucial for value)
    # Avoid division by zero by adding a small epsilon or ensuring denominator >= 1
    total_rooms = df['bedrooms'] + df['bathrooms'] + df['guestroom'].apply(lambda x: 1 if x == 'yes' else 0)
    df['area_per_room'] = df['area'] / total_rooms

    # Feature 2: Luxury Score (Combine yes/no amenities into a single rank)
    df['luxury_score'] = (
            df['mainroad'].apply(lambda x: 1 if x == 'yes' else 0) +
            df['guestroom'].apply(lambda x: 1 if x == 'yes' else 0) +
            df['basement'].apply(lambda x: 1 if x == 'yes' else 0) +
            df['hotwaterheating'].apply(lambda x: 1 if x == 'yes' else 0) +
            df['airconditioning'].apply(lambda x: 1 if x == 'yes' else 0) * 2 +  # AC is worth double points
            df['prefarea'].apply(lambda x: 1 if x == 'yes' else 0)
    )

    # Feature 3: Price per bedroom proxy (Bedroom density)
    df['bed_bath_ratio'] = df['bedrooms'] / df['bathrooms']

    return df


def build_model():
    # Update feature lists to include NEW columns
    categorical_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating',
                            'airconditioning', 'prefarea', 'furnishingstatus']

    # Added 'area_per_room', 'luxury_score', 'bed_bath_ratio'
    numerical_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking',
                          'area_per_room', 'luxury_score', 'bed_bath_ratio']

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Balanced Settings (better for small datasets than the Aggressive ones)
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42
        ))
    ])
    return model


def visualize_model_performance(model, X_test, y_test_original, predictions):
    sns.set(style="whitegrid")

    # 1. Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_original, predictions, alpha=0.6, color='b')
    plt.plot([y_test_original.min(), y_test_original.max()],
             [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
    plt.title('Actual vs Predicted Prices (Enriched)')
    plt.tight_layout()
    plt.show()

    # 2. Feature Importance (check if new features are being used)
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    importances = model.named_steps['regressor'].feature_importances_
    feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_df = feat_df.sort_values(by='Importance', ascending=False).head(10)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_df, palette='viridis')
    plt.title('Top 10 Feature Importances (Note new features!)')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = load_data()

    # 1. Enrich Data (Create new columns BEFORE splitting)
    df_enriched = enrich_data(df)

    X = df_enriched.drop('price', axis=1)

    # Log transform target (Still a good idea)
    y = np.log1p(df_enriched['price'])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training with Feature Engineering...")
    model = build_model()
    model.fit(X_train, y_train)

    # Evaluate
    log_predictions = model.predict(X_test)
    predictions = np.expm1(log_predictions)
    y_test_original = np.expm1(y_test)

    mae = mean_absolute_error(y_test_original, predictions)
    r2 = r2_score(y_test_original, predictions)

    print(f"Model Performance:")
    print(f"Mean Absolute Error: ${mae:,.2f}")
    print(f"Accuracy (R² Score): {r2:.2%}")

    visualize_model_performance(model, X_test, y_test_original, predictions)

    # --- Handling User Input ---
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

        # CRITICAL: Must apply the SAME enrichment to the user input
        input_enriched = enrich_data(input_df)

        # Predict
        log_price = model.predict(input_enriched)[0]
        real_price = np.expm1(log_price)

        print(f"\nEstimated Price: ${real_price:,.2f}")

    except ValueError as e:
        print(f"\nError in prediction: {e}")