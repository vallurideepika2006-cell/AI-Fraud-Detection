
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Create folder for charts
Path("reports").mkdir(exist_ok=True)

# Set chart style
sns.set_theme(style="whitegrid")

# Chart 1: Legitimate vs fraudulent transactions
class_counts = df["Class"].value_counts().sort_index()

plt.figure(figsize=(8, 5))
sns.barplot(
    x=["Legitimate", "Fraudulent"],
    y=class_counts.values
)
plt.title("Legitimate vs Fraudulent Transactions")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")
plt.tight_layout()
plt.savefig("reports/transaction_classes.png", dpi=300)
plt.show()

# Chart 2: Transaction amount distribution
plt.figure(figsize=(8, 5))
sns.histplot(
    data=df,
    x="Amount",
    bins=50
)
plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Number of Transactions")
plt.tight_layout()
plt.savefig("reports/transaction_amounts.png", dpi=300)
plt.show()

# Chart 3: Compare transaction amounts by class
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df,
    x="Class",
    y="Amount",
    showfliers=False
)
plt.title("Transaction Amounts by Class")
plt.xlabel("Class (0 = Legitimate, 1 = Fraudulent)")
plt.ylabel("Transaction Amount")
plt.tight_layout()
plt.savefig("reports/amounts_by_class.png", dpi=300)
plt.show()

print("All charts created successfully!")
print("Check the reports folder for the PNG files.")
