import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
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


def remove_outliers(df, column='price'):
    # Uses the IQR (Interquartile Range) method.
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    # Define bounds (1.5 is standard, we use 2.0 to be safer and not delete valid expensive houses)
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    initial_rows = len(df)
    df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].copy()
    rows_removed = initial_rows - len(df_clean)

    print(f"--- Outlier Removal ---")
    print(f"Removed {rows_removed} outliers based on {column}.")
    print(f"Data shape reduced from {initial_rows} to {len(df_clean)}.\n")

    return df_clean


def enrich_data(df):
    df = df.copy()

    # Ratios (Standard)
    is_guestroom = df['guestroom'].apply(lambda x: 1 if x == 'yes' else 0)
    total_rooms = df['bedrooms'] + df['bathrooms'] + is_guestroom
    df['area_per_room'] = df['area'] / total_rooms.replace(0, 1)
    df['bed_bath_ratio'] = df['bedrooms'] / df['bathrooms']

    # Interaction: Area * Stories (Vertical Space) - Often helps models
    df['vertical_space'] = df['area'] * df['stories']

    return df


def build_ensemble_model():
    categorical_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating',
                            'airconditioning', 'prefarea', 'furnishingstatus']

    numerical_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking',
                          'area_per_room', 'bed_bath_ratio', 'vertical_space']

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Model 1: XGBoost (The gradient booster)
    xgb = XGBRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42
    )

    # Model 2: Random Forest (The stabilizer)
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=15, n_jobs=-1, random_state=42
    )

    # Voting Regressor: Averages the predictions of both models
    ensemble = VotingRegressor(
        estimators=[('xgb', xgb), ('rf', rf)],
        weights=[0.6, 0.4]  # Trust XGBoost slightly more (60/40 split)
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', ensemble)
    ])

    return model


# --- NEW DASHBOARD FUNCTION (Visualization Only) ---
def plot_performance_dashboard(model, X_test, y_test, preds):
    sns.set(style="whitegrid")

    # Create a 2x2 grid of plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Housing Price Model Diagnostics', fontsize=18, weight='bold')

    # --- Chart 1: Actual vs Predicted (Accuracy) ---
    axes[0, 0].scatter(y_test, preds, alpha=0.5, color='#1f77b4', edgecolor='k', s=50)
    min_val = min(y_test.min(), preds.min())
    max_val = max(y_test.max(), preds.max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, label='Perfect Prediction')
    axes[0, 0].set_title('Actual vs Predicted Prices', fontsize=14)
    axes[0, 0].set_xlabel('Actual Price ($)')
    axes[0, 0].set_ylabel('Predicted Price ($)')
    axes[0, 0].legend()

    # --- Chart 2: Residual Plot (Bias Check) ---
    residuals = y_test - preds
    axes[0, 1].scatter(preds, residuals, alpha=0.5, color='#ff7f0e', edgecolor='k', s=50)
    axes[0, 1].axhline(y=0, color='black', linestyle='--', lw=2)
    axes[0, 1].set_title('Residuals (Errors) vs Predictions', fontsize=14)
    axes[0, 1].set_xlabel('Predicted Price ($)')
    axes[0, 1].set_ylabel('Error ($)')

    # --- Chart 3: Feature Importance (Logic Check) ---
    # We must access the internal models of the VotingRegressor to get importance
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()

    # Get importance from XGBoost (estimator 0) and Random Forest (estimator 1)
    xgb_imp = model.named_steps['regressor'].estimators_[0].feature_importances_
    rf_imp = model.named_steps['regressor'].estimators_[1].feature_importances_

    # Weighted Average (0.6 for XGB, 0.4 for RF)
    avg_imp = (xgb_imp * 0.6) + (rf_imp * 0.4)

    feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': avg_imp})
    feat_df = feat_df.sort_values(by='Importance', ascending=False).head(10)

    sns.barplot(x='Importance', y='Feature', data=feat_df, ax=axes[1, 0], palette='viridis')
    axes[1, 0].set_title('Top 10 Factors Influencing Price', fontsize=14)
    axes[1, 0].set_xlabel('Relative Importance')

    # --- Chart 4: Error Distribution (Normality Check) ---
    sns.histplot(residuals, kde=True, ax=axes[1, 1], color='#9467bd', bins=30)
    axes[1, 1].axvline(x=0, color='black', linestyle='--')
    axes[1, 1].set_title('Distribution of Prediction Errors', fontsize=14)
    axes[1, 1].set_xlabel('Error ($)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for main title
    plt.show()


if __name__ == "__main__":
    df = load_data()

    # 1. Remove Outliers (Crucial step for higher R2)
    df_clean = remove_outliers(df, 'price')

    # 2. Enrich
    df_enriched = enrich_data(df_clean)

    # 3. Prepare Split
    X = df_enriched.drop('price', axis=1)
    y = np.log1p(df_enriched['price'])  # Log transform is still best practice

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Ensemble Model (XGBoost + Random Forest)...")
    model = build_ensemble_model()
    model.fit(X_train, y_train)

    # 4. Evaluate
    log_predictions = model.predict(X_test)
    predictions = np.expm1(log_predictions)
    y_test_original = np.expm1(y_test)

    mae = mean_absolute_error(y_test_original, predictions)
    r2 = r2_score(y_test_original, predictions)

    print(f"Model Performance:")
    print(f"Mean Absolute Error: ${mae:,.2f}")
    print(f"Accuracy (R² Score): {r2:.2%}")

    # 5. Visualization (UPDATED: Now calls the Dashboard function)
    print("\nGenerating Presentation Dashboard...")
    plot_performance_dashboard(model, X_test, y_test_original, predictions)

    # 6. User Prediction
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
        input_enriched = enrich_data(input_df)
        log_price = model.predict(input_enriched)[0]
        print(f"\nEstimated Price: ${np.expm1(log_price):,.2f}")
    except Exception as e:
        print(f"Prediction Error: {e}")