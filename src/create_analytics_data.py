
import json
from pathlib import Path

import pandas as pd

# Project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "creditcard.csv"
OUTPUT_FILE = PROJECT_DIR / "frontend" / "analytics_data.json"


def main():
    print("Loading transaction dataset...")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    required_columns = {"Amount", "Class"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Dataset must contain Amount and Class columns."
        )

    # Calculate transaction counts
    total = len(df)
    fraud_count = int((df["Class"] == 1).sum())
    legitimate_count = int((df["Class"] == 0).sum())

    fraud_percentage = (
        (fraud_count / total) * 100 if total else 0
    )

    # Create transaction amount ranges
    amount_ranges = [
        ("0-25", 0, 25),
        ("25-50", 25, 50),
        ("50-100", 50, 100),
        ("100-250", 100, 250),
        ("250-500", 250, 500),
        ("500+", 500, float("inf")),
    ]

    amount_distribution = []

    for label, lower, upper in amount_ranges:
        if upper == float("inf"):
            selected = df[df["Amount"] >= lower]
        else:
            selected = df[
                (df["Amount"] >= lower)
                & (df["Amount"] < upper)
            ]

        amount_distribution.append({
            "range": label,
            "legitimate": int(
                (selected["Class"] == 0).sum()
            ),
            "fraud": int(
                (selected["Class"] == 1).sum()
            ),
        })

    analytics = {
        "total_transactions": int(total),
        "legitimate_transactions": legitimate_count,
        "fraudulent_transactions": fraud_count,
        "fraud_percentage": round(fraud_percentage, 4),
        "amount_distribution": amount_distribution,
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(analytics, file, indent=4)

    print("\nAnalytics data generated successfully!")
    print(f"Total transactions: {total:,}")
    print(f"Legitimate transactions: {legitimate_count:,}")
    print(f"Fraudulent transactions: {fraud_count:,}")
    print(f"Fraud percentage: {fraud_percentage:.4f}%")
    print(f"\nOutput file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
