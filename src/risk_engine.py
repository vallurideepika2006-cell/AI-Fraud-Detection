
import joblib
import pandas as pd

from pathlib import Path

MODEL_PATH = Path("models/random_forest.joblib")
SCALER_PATH = Path("models/amount_scaler.joblib")
TEST_PATH = Path("data/processed/test.csv")


class FraudRiskEngine:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Random Forest model not found. Train the model first."
            )

        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)

        # The exact input feature order used during training
        self.features = list(
            pd.read_csv(TEST_PATH, nrows=1)
            .drop(columns=["Class"])
            .columns
        )

    def predict(self, transaction):
        """
        transaction must contain Time, V1-V28, and Amount.
        Amount must be in its original, unscaled units.
        """

        missing = set(self.features) - set(transaction.keys())

        if missing:
            raise ValueError(
                f"Missing transaction features: {sorted(missing)}"
            )

        # Arrange features in the correct order
        row = pd.DataFrame(
            [[transaction[feature] for feature in self.features]],
            columns=self.features
        )

        # Apply the same scaler used during training
        row["Amount"] = self.scaler.transform(
            row[["Amount"]]
        ).ravel()

        # Obtain predicted fraud probability
        probability = float(
            self.model.predict_proba(row)[0, 1]
        )

        # Convert probability to a score from 0 to 100
        risk_score = round(probability * 100, 2)

        # Example risk bands; these are initial thresholds
        if risk_score < 30:
            risk_level = "Low"
        elif risk_score < 70:
            risk_level = "Medium"
        else:
            risk_level = "High"

        return {
            "fraud_probability": round(probability, 4),
            "risk_score": risk_score,
            "risk_level": risk_level,
            "prediction": (
                "Fraud" if risk_score >= 50 else "Legitimate"
            )
        }


if __name__ == "__main__":
    engine = FraudRiskEngine()

    # Use an actual held-out transaction as a demonstration.
    # Exclude its Class label so it isn't given to the model.
    test_df = pd.read_csv(TEST_PATH)
    sample = test_df.drop(columns=["Class"]).iloc[0].to_dict()

    result = engine.predict(sample)

    print("\n--- FRAUD RISK ASSESSMENT ---")
    for key, value in result.items():
        print(f"{key}: {value}")
