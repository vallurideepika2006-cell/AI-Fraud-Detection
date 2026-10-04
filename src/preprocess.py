
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# File paths
DATA_PATH = Path("data/creditcard.csv")
OUTPUT_DIR = Path("data/processed")
MODEL_DIR = Path("models")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)

# Check required target column
if "Class" not in df.columns:
    raise ValueError("The dataset must contain a 'Class' column.")

# Remove duplicate rows
duplicates = df.duplicated().sum()
print("Duplicate rows found:", duplicates)

df = df.drop_duplicates().copy()
print("Shape after removing duplicates:", df.shape)

# Check missing values
missing = df.isnull().sum().sum()
print("Missing values:", missing)

if missing > 0:
    df = df.dropna().copy()
    print("Shape after removing missing values:", df.shape)

# Separate features and target
X = df.drop(columns=["Class"])
y = df["Class"].astype(int)

print("\nClass distribution:")
print(y.value_counts())

# Split before fitting the scaler to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Scale Amount using training data only.
# Time and V1-V28 are left unchanged in this first version.
scaler = StandardScaler()

X_train = X_train.copy()
X_test = X_test.copy()

X_train["Amount"] = scaler.fit_transform(X_train[["Amount"]]).ravel()
X_test["Amount"] = scaler.transform(X_test[["Amount"]]).ravel()

# Save datasets
train_df = X_train.copy()
train_df["Class"] = y_train

test_df = X_test.copy()
test_df["Class"] = y_test

train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

# Save scaler for future predictions
joblib.dump(scaler, MODEL_DIR / "amount_scaler.joblib")

print("\nPreprocessing completed successfully!")
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
print("Training data saved to:", OUTPUT_DIR / "train.csv")
print("Testing data saved to:", OUTPUT_DIR / "test.csv")
print("Scaler saved to:", MODEL_DIR / "amount_scaler.joblib")
