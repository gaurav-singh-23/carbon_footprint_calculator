import pandas as pd

# Load your dataset
df = pd.read_csv("data/Carbon_Emission.csv")

# Basic info
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:\n", df.head())

# Check missing values
print("\nMissing values per column:\n", df.isna().sum())

# Quick statistics
print("\nSummary statistics:\n", df.describe(include="all"))
