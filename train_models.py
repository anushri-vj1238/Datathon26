from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from model_io import save_model

DATA_DIR = Path(__file__).resolve().parent / "data"
DATASET_PATH = DATA_DIR / "merged_dataset.csv"
MODEL_PATH = DATA_DIR / "model.joblib"

FEATURE_COLUMNS = ["tempo", "energy", "valence", "cognitive_load_final"]
TARGET_COLUMN = "productivity"


def train_models(dataset_path: Path = DATASET_PATH, model_path: Path = MODEL_PATH) -> None:
    df = pd.read_csv(dataset_path)

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in merged dataset: {missing}")

    data = df[required_columns].apply(pd.to_numeric, errors="coerce").dropna().copy()
    X = data[FEATURE_COLUMNS].astype(float)
    y = data[TARGET_COLUMN].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred_linear = linear_model.predict(X_test)
    linear_r2 = r2_score(y_test, y_pred_linear)
    print(f"Linear Regression R² (test): {linear_r2:.4f}")

    rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    rf_r2 = r2_score(y_test, y_pred_rf)
    print(f"Random Forest R² (test): {rf_r2:.4f}")

    importances = pd.Series(rf_model.feature_importances_, index=FEATURE_COLUMNS)
    print("Random Forest feature importances:")
    for feature, importance in importances.sort_values(ascending=False).items():
        print(f"  {feature}: {importance:.4f}")

    save_model(rf_model, model_path)
    print(f"Saved Random Forest model to: {model_path}")


if __name__ == "__main__":
    train_models()
