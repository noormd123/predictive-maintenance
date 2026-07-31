# Train XGBoost model with MLflow tracking; commit the best model into deployment/
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Resolve paths relative to script location (local dev and CI)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "deployment"
MODEL_FILENAME = "best_predictive_maintenance_model_v1.joblib"
MODEL_PATH = MODEL_DIR / MODEL_FILENAME
MANIFEST_PATH = Path(__file__).resolve().parent / "model_upload_manifest.json"

TARGET_COL = "Engine Condition"
RANDOM_STATE = 42

# MLflow tracking for production and CI/CD runs
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("predictive-maintenance-experiment")

# Step 1: Load train and test data (Xtrain/Xtest/ytrain/ytest come from the previous job's artifact)
Xtrain = pd.read_csv(DATA_DIR / "Xtrain.csv")
Xtest = pd.read_csv(DATA_DIR / "Xtest.csv")
ytrain = pd.read_csv(DATA_DIR / "ytrain.csv")[TARGET_COL]
ytest = pd.read_csv(DATA_DIR / "ytest.csv")[TARGET_COL]
print("Train and test data loaded successfully.")

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

    # Step 5: Save the best model next to app.py so the Streamlit app can load it directly
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")
    print(f"Model saved to {MODEL_PATH}")

    manifest = {
        "uploaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_file": MODEL_FILENAME,
        "best_params": grid_search.best_params_,
        "test_f1_score_class_1": test_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest written to {MANIFEST_PATH}")

print("Model building complete.")
