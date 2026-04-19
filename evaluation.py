import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

BASE_DIR = Path(__file__).parent
PROCESSED_DIR = BASE_DIR #/ "data" / "processed"

def evaluate_clf(x_test,y_test,run_id):

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    preds = model.predict(x_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="macro")
    rec = recall_score(y_test, preds, average="macro")
    f1 = f1_score(y_test, preds, average="macro")

    mlflow.set_tracking_uri("file:./mlruns")
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

    print(f"Evaluation completed | Accuracy = {acc:.3f}")

    return acc, prec, f1

def evaluate_reg(x_test, y_test, run_id):
    # Load model regresi
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    mlflow.set_tracking_uri("file:./mlruns")
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("eval_MAE", mae)
        mlflow.log_metric("eval_RMSE", rmse)
        mlflow.log_metric("eval_R2", r2)

    print(f"Evaluation completed | R2 = {r2:.3f}")
    
    return mae, rmse, r2