import pandas as pd
from data_ingestion import ingest_data
from train import train_model_clf, train_model_reg
from evaluation import evaluate_clf, evaluate_reg
from sklearn.model_selection import train_test_split

# Konfigurasi
SEED = 42
TEST_SIZE = 0.2
F1_THRESHOLD = 0.80

def run_pipeline():
    print(">>> Step 1: Data Ingestion")
    ingest_data()
    
    # Load data hasil ingestion
    df = pd.read_csv("ingested/B.csv")

    # Feature Engineering
    df['avg_academic'] = (df['ssc_percentage'] + df['hsc_percentage'] + df['degree_percentage']) / 3
    df['total_skill'] = (df['technical_skill_score'] + df['soft_skill_score']) / 2
    df['experience_idx'] = (df['internship_count'] * 2) + df['live_projects'] + (df['work_experience_months'] / 3) + df['certifications']
    df['has_backlogs'] = (df['backlogs'] > 0).astype(int)

    # --- MODEL 1: CLASSIFICATION (Semua Data) ---
    X_clf = df.drop(columns=["student_id", "placement_status", "salary_package_lpa"])
    y_clf = df["placement_status"]
    
    x_train_clf, x_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_clf, test_size=TEST_SIZE, random_state=SEED, stratify=y_clf
    )

    print("\n>>> Step 2.1: Training & Eval Classification")
    run_id_clf = train_model_clf(x_train_clf, y_train_clf)
    acc, prec, f1 = evaluate_clf(x_test_clf, y_test_clf, run_id_clf)

    # --- MODEL 2: REGRESSION (Hanya yang Placed) ---
    df_placed = df[df['placement_status'] == 1].copy()
    X_reg = df_placed.drop(columns=["student_id", "placement_status", "salary_package_lpa"])
    y_reg = df_placed["salary_package_lpa"]

    x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=TEST_SIZE, random_state=SEED
    )

    print("\n>>> Step 2.2: Training & Eval Regression")
    run_id_reg = train_model_reg(x_train_reg, y_train_reg)
    mae, rmse, r2 = evaluate_reg(x_test_reg, y_test_reg, run_id_reg)

    print("\n>>> Step 3: Deployment Decision")
    if f1 >= F1_THRESHOLD:
        print(f"✅ Model APPROVED! (F1: {f1:.4f} >= {F1_THRESHOLD})")
    else:
        print(f"❌ Model REJECTED! (F1: {f1:.4f} < {F1_THRESHOLD})")

if __name__ == "__main__":
    run_pipeline()