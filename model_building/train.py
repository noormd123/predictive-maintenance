# Train XGBoost model with MLflow tracking; register best model on Hugging Face Hub
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import xgboost as xgb
from huggingface_hub import CommitOperationAdd, HfApi, create_commit, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
from sklearn.compose import make_column_transformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Resolve paths relative to script location (local dev and CI)
MODEL_DIR = Path(__file__).resolve().parent
MODEL_FILENAME = "best_predictive_maintenance_model_v1.joblib"
MODEL_PATH = MODEL_DIR / MODEL_FILENAME
MANIFEST_FILENAME = "model_upload_manifest.json"
MANIFEST_PATH = MODEL_DIR / MANIFEST_FILENAME

os.environ.setdefault("MLFLOW_GIT_DISABLE", "1")

# MLflow tracking for production and CI/CD runs
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("predictive-maintenance-experiment")

# Hugging Face dataset and model repositories
DATASET_REPO = "noormd100/predictive-maintenance-data"
MODEL_REPO = "noormd100/predictive-maintenance-model"
TARGET_COL = "Engine Condition"
RANDOM_STATE = 42

api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Load train and test data from Hugging Face Hub
Xtrain = pd.read_csv(f"hf://datasets/{DATASET_REPO}/Xtrain.csv")
Xtest = pd.read_csv(f"hf://datasets/{DATASET_REPO}/Xtest.csv")
ytrain = pd.read_csv(f"hf://datasets/{DATASET_REPO}/ytrain.csv")[TARGET_COL]
ytest = pd.read_csv(f"hf://datasets/{DATASET_REPO}/ytest.csv")[TARGET_COL]
print("Train and test data loaded successfully from Hugging Face Hub.")

feature_cols = Xtrain.columns.tolist()

# Step 2: Define model, pipeline, and hyperparameter grid
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

preprocessor = make_column_transformer((StandardScaler(), feature_cols))
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=RANDOM_STATE,
    eval_metric="logloss",
)
param_grid = {
    "xgbclassifier__n_estimators": [50, 100],
    "xgbclassifier__max_depth": [2, 3, 4],
    "xgbclassifier__learning_rate": [0.05, 0.1],
}
model_pipeline = make_pipeline(preprocessor, xgb_model)
print("XGBoost model and hyperparameter grid defined.")

# Step 3: Tune model with GridSearchCV and log all tuned parameters in MLflow
with mlflow.start_run():
    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=1,
    )
    grid_search.fit(Xtrain, ytrain)
    print("Hyperparameter tuning complete.")

    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        mean_score = results["mean_test_score"][i]
        if math.isnan(mean_score):
            continue
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", results["std_test_score"][i])

    mlflow.log_params(grid_search.best_params_)
    print(f"Best parameters: {grid_search.best_params_}")

    # Step 4: Evaluate model performance on train and test sets
    best_model = grid_search.best_estimator_
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    mlflow.log_metrics(
        {
            "train_accuracy": train_report["accuracy"],
            "train_precision": train_report["1"]["precision"],
            "train_recall": train_report["1"]["recall"],
            "train_f1-score": train_report["1"]["f1-score"],
            "test_accuracy": test_report["accuracy"],
            "test_precision": test_report["1"]["precision"],
            "test_recall": test_report["1"]["recall"],
            "test_f1-score": test_report["1"]["f1-score"],
        }
    )

    print("\nTrain Classification Report:")
    print(classification_report(ytrain, y_pred_train))
    print("\nTest Classification Report:")
    print(classification_report(ytest, y_pred_test))

    # Step 5: Save best model locally and register on Hugging Face model hub
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")
    print(f"Model saved locally as {MODEL_PATH}")

    try:
        api.repo_info(repo_id=MODEL_REPO, repo_type="model")
        print(f"Model repo '{MODEL_REPO}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Model repo '{MODEL_REPO}' not found. Creating new model repo...")
        create_repo(repo_id=MODEL_REPO, repo_type="model", private=False)
        print(f"Model repo '{MODEL_REPO}' created.")

    uploaded_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "uploaded_at_utc": uploaded_at,
        "model_file": MODEL_FILENAME,
        "best_params": grid_search.best_params_,
        "test_f1_score_class_1": test_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    create_commit(
        repo_id=MODEL_REPO,
        repo_type="model",
        operations=[
            CommitOperationAdd(
                path_in_repo=MODEL_FILENAME,
                path_or_fileobj=str(MODEL_PATH),
            ),
            CommitOperationAdd(
                path_in_repo=MANIFEST_FILENAME,
                path_or_fileobj=str(MANIFEST_PATH),
            ),
        ],
        commit_message=f"Upload retrained model ({uploaded_at})",
        token=os.getenv("HF_TOKEN"),
    )
    print(f"Model and manifest uploaded to {MODEL_REPO} at {uploaded_at}")

print("Model building complete.")
