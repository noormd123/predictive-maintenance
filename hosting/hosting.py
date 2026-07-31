# Upload deployment files to Hugging Face Space
#
# NOTE: this is a best-effort script. HF Spaces hosting has hit an account
# license/limit, so the GitHub Actions job that runs this is allowed to fail
# (continue-on-error) without breaking the pipeline. The live app is deployed
# separately via Streamlit Community Cloud, which reads deployment/app.py
# directly from this repo.
import os
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError

# Resolve deployment folder relative to script location (local dev and CI)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEPLOYMENT_FOLDER = PROJECT_ROOT / "deployment"

# Hugging Face Space repository
SPACE_REPO = "noormd100/predictive-maintenance-prediction"

# Authenticate with HF_TOKEN and upload deployment folder
api = HfApi(token=os.getenv("HF_TOKEN"))

try:
    api.repo_info(repo_id=SPACE_REPO, repo_type="space")
    print(f"Space repo '{SPACE_REPO}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space repo '{SPACE_REPO}' not found. Creating new Space (Docker SDK)...")
    api.create_repo(repo_id=SPACE_REPO, repo_type="space", space_sdk="docker", private=False)
    print(f"Space repo '{SPACE_REPO}' created.")

api.upload_folder(
    folder_path=str(DEPLOYMENT_FOLDER),
    repo_id=SPACE_REPO,
    repo_type="space",
    path_in_repo="",
)
print(f"Deployment files uploaded to {SPACE_REPO}")
