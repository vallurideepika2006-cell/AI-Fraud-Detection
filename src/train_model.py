
import pandas as pd
import joblib
import time

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score
)

# Paths
TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

print("Loading prepared datasets...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df.drop(columns=["Class"])
y_train = train_df["Class"]

X_test = test_df.drop(columns=["Class"])
y_test = test_df["Class"]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Training fraud cases:", int(y_train.sum()))
print("Testing fraud cases:", int(y_test.sum()))

# Define models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
}

results = []

for name, model in models.items():
    print(f"\n{'=' * 60}")
    print("Training:", name)

    start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start

    # Predict classes and fraud probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    precision = report["1"]["precision"]
    recall = report["1"]["recall"]
    f1 = report["1"]["f1-score"]
    average_precision = average_precision_score(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print(f"Fraud precision: {precision:.4f}")
    print(f"Fraud recall: {recall:.4f}")
    print(f"Fraud F1-score: {f1:.4f}")
    print(f"PR-AUC (average precision): {average_precision:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Training time: {training_time:.2f} seconds")

    # Save each trained model
    filename = (
        "logistic_regression.joblib"
        if name == "Logistic Regression"
        else "random_forest.joblib"
    )

    joblib.dump(model, MODEL_DIR / filename)

    results.append({
        "Model": name,
        "Fraud Precision": precision,
        "Fraud Recall": recall,
        "Fraud F1": f1,
        "PR-AUC": average_precision,
        "ROC-AUC": roc_auc,
        "Training Time (seconds)": round(training_time, 2)
    })

# Save comparison table
results_df = pd.DataFrame(results)
results_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print(results_df.to_string(index=False))
print("\nSaved comparison to reports/model_comparison.csv")
print("Model training completed successfully!")
