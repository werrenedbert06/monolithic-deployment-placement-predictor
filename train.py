import joblib
import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge

SEED = 42
# 1. Definisi Kolom
FE_COLS = ["avg_academic", "total_skill", "experience_idx", "has_backlogs"]
NUM_COLS = ["ssc_percentage", "hsc_percentage", "degree_percentage", "cgpa", 
            "entrance_exam_score", "technical_skill_score", "soft_skill_score", 
            "internship_count", "live_projects", "work_experience_months", 
            "certifications", "attendance_percentage", "backlogs"] + FE_COLS
NOM_COLS = ["gender", "extracurricular_activities"]

def train_model_clf(x_train, y_train):
    os.makedirs("artifacts", exist_ok=True)

    # 2. Setup Preprocessor
    preprocess = ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), NOM_COLS)
    ], remainder="drop")

    # 3. Setup Pipeline Utuh
    placement_pred = Pipeline([
        ("preprocessing", preprocess),
        ("classifier", RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1))
    ])

    # ===== MLflow tracking =====
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Placement Prediction")

    with mlflow.start_run() as run:
        # log parameters
        mlflow.log_param("model", "RandomForestClassifier")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("random_state", SEED)

        # train
        placement_pred.fit(x_train, y_train)

        # save and log model
        joblib.dump(placement_pred, "artifacts/placement_prediction_pipeline.pkl")
        mlflow.sklearn.log_model(placement_pred, artifact_path="model")

    return run.info.run_id

def train_model_reg(x_train, y_train):
    os.makedirs("artifacts", exist_ok=True)

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), NOM_COLS)
    ])

    # 3. Setup Pipeline Utuh pake RIDGE REGRESSION
    salary_pred = Pipeline([
        ("preprocessing", preprocessor),
        ("regressor", Ridge(alpha=1.0, random_state=SEED))
    ])

    # ===== MLflow tracking =====
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Salary Prediction")

    with mlflow.start_run() as run:
        # log parameters (Pattern Code 2)
        mlflow.log_param("model", "Ridge Regression")
        mlflow.log_param("alpha", 1.0)
        mlflow.log_param("random_state", SEED)

        # train (y_train di-ravel supaya tidak kena warning)
        salary_pred.fit(x_train, y_train.values.ravel())

        joblib.dump(salary_pred, "artifacts/salary_prediction_pipeline.pkl")
        mlflow.sklearn.log_model(salary_pred, artifact_path="model")

    return run.info.run_id
