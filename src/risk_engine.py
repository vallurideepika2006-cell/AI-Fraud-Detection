import joblib
import pandas as pd
import shap
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_DIR / "models" / "random_forest.joblib"
SCALER_PATH = PROJECT_DIR / "models" / "amount_scaler.joblib"
TEST_DATA_PATH = PROJECT_DIR / "data" / "processed" / "test.csv"


class FraudRiskEngine:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)

        test_data = pd.read_csv(TEST_DATA_PATH)
        self.feature_names = [
            column for column in test_data.columns
            if column != "Class"
        ]

        self.explainer = shap.TreeExplainer(self.model)

    def predict(self, transaction):
        transaction_df = pd.DataFrame(
            [transaction],
            columns=self.feature_names
        )

        # Apply the same Amount scaling used during training.
        transaction_df["Amount"] = self.scaler.transform(
            transaction_df[["Amount"]]
        ).ravel()

        probabilities = self.model.predict_proba(transaction_df)[0]
        classes = list(self.model.classes_)

        fraud_index = classes.index(1)
        fraud_probability = float(probabilities[fraud_index])

        risk_score = round(fraud_probability * 100, 2)

        if risk_score < 30:
            risk_level = "Low"
        elif risk_score < 70:
            risk_level = "Medium"
        else:
            risk_level = "High"

        prediction = "Fraud" if risk_score >= 50 else "Legitimate"

        # Explain the model's fraud probability using SHAP.
        shap_values = self.explainer.shap_values(transaction_df)

        # Handle SHAP output formats for binary classification.
        if isinstance(shap_values, list):
            fraud_shap_values = shap_values[fraud_index][0]
        elif getattr(shap_values, "ndim", 0) == 3:
            fraud_shap_values = shap_values[0, :, fraud_index]
        else:
            fraud_shap_values = shap_values[0]

        feature_importance = sorted(
            [
                {
                    "feature": feature,
                    "impact": round(float(value), 6)
                }
                for feature, value in zip(
                    self.feature_names,
                    fraud_shap_values
                )
            ],
            key=lambda item: abs(item["impact"]),
            reverse=True
        )

        top_features = feature_importance[:5]

        explanations = [
            {
                "feature": item["feature"],
                "impact": item["impact"],
                "effect": (
                    "increases fraud probability"
                    if item["impact"] > 0
                    else "decreases fraud probability"
                    if item["impact"] < 0
                    else "has little or no effect"
                )
            }
            for item in top_features
        ]

        return {
            "prediction": prediction,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "explanation": explanations
        }