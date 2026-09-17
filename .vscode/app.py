import os
from typing import Any, Dict, List

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

app = Flask(__name__)

DATA_PATH = os.getenv("SUPPLY_CHAIN_DATA_PATH", r"E:\dynamic_supply_chain_logistics_dataset.csv")
MODEL_DIR = os.getenv("MODEL_DIR", "model_artifacts")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "supply_chain_classifier.joblib")
REGRESSOR_PATH = os.path.join(MODEL_DIR, "supply_chain_regressor.joblib")


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_columns = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

    transformers = []

    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )

    if not transformers:
        raise ValueError("No usable feature columns found after removing target and timestamp columns.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def train_models() -> tuple:
    df = pd.read_csv(DATA_PATH)

    if "risk_classification" not in df.columns or "delivery_time_deviation" not in df.columns:
        raise ValueError("Dataset must contain 'risk_classification' and 'delivery_time_deviation' columns.")

    features = df.drop(columns=["timestamp", "risk_classification", "delivery_time_deviation"], errors="ignore")
    if features.empty:
        raise ValueError("No feature columns left after dropping target and timestamp columns.")

    target_classification = df["risk_classification"]
    target_regression = df["delivery_time_deviation"]

    X_train, _, y_train_clf, _, = train_test_split(
        features, target_classification, test_size=0.2, random_state=42, stratify=target_classification
    )
    _, X_test, _, y_test_clf = train_test_split(
        features, target_classification, test_size=0.2, random_state=42, stratify=target_classification
    )

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        features, target_regression, test_size=0.2, random_state=42
    )

    classifier_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("model", RandomForestClassifier(random_state=42, n_estimators=200)),
        ]
    )

    regressor_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("model", RandomForestRegressor(random_state=42, n_estimators=200)),
        ]
    )

    classifier_pipeline.fit(X_train, y_train_clf)
    regressor_pipeline.fit(X_train_reg, y_train_reg)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(classifier_pipeline, CLASSIFIER_PATH)
    joblib.dump(regressor_pipeline, REGRESSOR_PATH)

    return classifier_pipeline, regressor_pipeline


def load_or_train_models() -> tuple:
    if os.path.exists(CLASSIFIER_PATH) and os.path.exists(REGRESSOR_PATH):
        classifier = joblib.load(CLASSIFIER_PATH)
        regressor = joblib.load(REGRESSOR_PATH)
        return classifier, regressor
    return train_models()


classifier_model, regressor_model = load_or_train_models()


def normalize_input(data: Any) -> pd.DataFrame:
    if isinstance(data, dict):
        if "features" in data:
            feature_data = data["features"]
            if isinstance(feature_data, list):
                return pd.DataFrame(feature_data)
            return pd.DataFrame([feature_data])
        if "data" in data:
            if isinstance(data["data"], list):
                return pd.DataFrame(data["data"])
            return pd.DataFrame([data["data"]])
        return pd.DataFrame([data])

    if isinstance(data, list):
        return pd.DataFrame(data)

    raise ValueError("Input should be a dictionary or list of dictionaries.")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Supply chain model API is running."})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)

    if not payload:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    try:
        df = normalize_input(payload)
        expected_columns = list(classifier_model.feature_names_in_)
        missing_columns = [col for col in expected_columns if col not in df.columns]

        if missing_columns:
            return jsonify({"error": f"Missing required fields: {missing_columns}"}), 400

        df = df[expected_columns]

        risk_prediction = classifier_model.predict(df)
        risk_probabilities = classifier_model.predict_proba(df)
        top_probability = risk_probabilities.max(axis=1)
        delivery_time_prediction = regressor_model.predict(df)

        results = []
        for i, risk_label in enumerate(risk_prediction):
            results.append(
                {
                    "risk_classification": str(risk_label),
                    "risk_probability": float(top_probability[i]),
                    "delivery_time_deviation": float(delivery_time_prediction[i]),
                }
            )

        if len(results) == 1:
            return jsonify(results[0])
        return jsonify({"predictions": results})

    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)