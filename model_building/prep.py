# Load, clean, feature-engineer, and split engine sensor data; upload splits to Hugging Face Hub
import os
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split

# Resolve paths relative to script location (local dev and CI)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Hugging Face dataset repository
REPO_ID = "noormd100/predictive-maintenance-data"
DATASET_PATH = f"hf://datasets/{REPO_ID}/engine_data.csv"
TARGET_COL = "Engine Condition"
RANDOM_STATE = 42
COOLANT_CAP = 100.0

api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Load dataset from Hugging Face Hub
df = pd.read_csv(DATASET_PATH)
print(f"Dataset loaded from Hugging Face: {df.shape[0]:,} rows x {df.shape[1]} columns")

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

DATA_DIR.mkdir(parents=True, exist_ok=True)

# Step 4: Save train and test splits locally
Xtrain.to_csv(DATA_DIR / "Xtrain.csv", index=False)
Xtest.to_csv(DATA_DIR / "Xtest.csv", index=False)
ytrain.to_csv(DATA_DIR / "ytrain.csv", index=False)
ytest.to_csv(DATA_DIR / "ytest.csv", index=False)
print(f"Saved split files to {DATA_DIR}/")

# Step 5: Upload split files back to Hugging Face dataset repo
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]
for filename in files:
    file_path = DATA_DIR / filename
    api.upload_file(
        path_or_fileobj=str(file_path),
        path_in_repo=filename,
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    print(f"Uploaded {file_path} to {REPO_ID}")

print("Data preparation complete.")
