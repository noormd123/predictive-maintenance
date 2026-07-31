# Register raw engine sensor dataset on Hugging Face Hub
import os
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# Resolve paths relative to script location (local dev and CI)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_PATH = DATA_DIR / "engine_data.csv"

# Hugging Face dataset repository
REPO_ID = "noormd100/predictive-maintenance-data"
REPO_TYPE = "dataset"

# Load the raw dataset and validate it before registering
df = pd.read_csv(RAW_PATH)

expected_columns = [
    "Engine rpm", "Lub oil pressure", "Fuel pressure",
    "Coolant pressure", "lub oil temp", "Coolant temp", "Engine Condition",
]
missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))
print("Engine Condition distribution:")
print(df["Engine Condition"].value_counts())

# Authenticate with HF_TOKEN from environment
api = HfApi(token=os.getenv("HF_TOKEN"))

# Create the dataset repo if it does not exist
try:
    api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
    print(f"Dataset repo '{REPO_ID}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset repo '{REPO_ID}' not found. Creating new dataset repo...")
    create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, private=False)
    print(f"Dataset repo '{REPO_ID}' created.")

# Upload the local data folder to Hugging Face Hub
api.upload_folder(
    folder_path=str(DATA_DIR),
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
)
print(f"Uploaded {DATA_DIR} to {REPO_ID}")
