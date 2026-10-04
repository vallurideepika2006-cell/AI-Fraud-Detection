
import pandas as pd

# Load the fraud detection dataset
df = pd.read_csv("data/creditcard.csv")

# 1. Display dataset dimensions
print("\n--- DATASET SHAPE ---")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# 2. Display the first five rows
print("\n--- FIRST FIVE TRANSACTIONS ---")
print(df.head())

# 3. Display column names and data types
print("\n--- DATASET INFORMATION ---")
df.info()

# 4. Check for missing values
print("\n--- MISSING VALUES ---")
print(df.isnull().sum().sum())

# 5. Check for duplicate rows
print("\n--- DUPLICATE ROWS ---")
print(df.duplicated().sum())

# 6. Count legitimate and fraudulent transactions
print("\n--- TRANSACTION CLASS COUNTS ---")
print(df["Class"].value_counts())

# 7. Calculate fraud percentages
print("\n--- TRANSACTION CLASS PERCENTAGES ---")
print((df["Class"].value_counts(normalize=True) * 100).round(4))

# 8. Analyze transaction amounts
print("\n--- TRANSACTION AMOUNT STATISTICS ---")
print(df["Amount"].describe())

print("\nEDA basic analysis completed successfully!")
