import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import os
import glob
import kagglehub

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
    categorical_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating',
                            'airconditioning', 'prefarea', 'furnishingstatus']
    numerical_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    return model


if __name__ == "__main__":
    df = load_data()

    X = df.drop('price', axis=1)
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    model = build_model()
    model.fit(X_train, y_train)

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

        predicted_price = model.predict(input_df)[0]
        print(f"\nEstimated Price: ${predicted_price:,.2f}")

    except ValueError:
        print("\nInvalid input. Check input fields.")