# Load, clean, feature-engineer, and split engine sensor data (train/test CSVs stay local)
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Resolve paths relative to script location (local dev and CI)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

TARGET_COL = "Engine Condition"
RANDOM_STATE = 42
COOLANT_CAP = 100.0

# Step 1: Load dataset from the repo
df = pd.read_csv(DATA_DIR / "engine_data.csv")
print(f"Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

# Step 2: Clean data and engineer features
extreme_count = (df["Coolant temp"] > COOLANT_CAP).sum()
df["Coolant temp"] = df["Coolant temp"].clip(upper=COOLANT_CAP)
print(f"Capped {extreme_count} Coolant temp values at {COOLANT_CAP} degrees Celsius")

df["temp_differential"] = df["Coolant temp"] - df["lub oil temp"]
print("Added feature: temp_differential")
print(f"Cleaned dataset shape: {df.shape}")

# Step 3: Split into features (X) and target (y), then train-test split (80/20)
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train size: {len(Xtrain):,}, Test size: {len(Xtest):,}")

# Step 4: Save train and test splits locally (passed between GitHub Actions jobs as an artifact)
Xtrain.to_csv(DATA_DIR / "Xtrain.csv", index=False)
Xtest.to_csv(DATA_DIR / "Xtest.csv", index=False)
ytrain.to_csv(DATA_DIR / "ytrain.csv", index=False)
ytest.to_csv(DATA_DIR / "ytest.csv", index=False)
print(f"Saved split files to {DATA_DIR}/")

print("Data preparation complete.")
