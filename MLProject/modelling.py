import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

# =========================
# Konfigurasi
# =========================
DATA_PATH = "dataset_mesin_membangun_sistem_machine_learning_preprocessing.csv"
EXPERIMENT_NAME = "Student Performance - Modelling"
RUN_ID_PATH = "run_id.txt"

def run_model():
    # =========================
    # MLflow setup
    # =========================
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("Training dimulai...")

    # =========================
    # Load dataset
    # =========================
    df = pd.read_csv(DATA_PATH)
    print("Dataset berhasil diload")

    # =========================
    # Feature & target
    # =========================
    X = df.drop(columns=["FinalGrade"])
    y = df["FinalGrade"]

    # =========================
    # Train-test split
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =========================
    # Training (SATU RUN SAJA)
    # =========================
    with mlflow.start_run(run_name="RandomForest-Baseline") as run:
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

        model.fit(X_train, y_train)

        # =========================
        # Evaluation
        # =========================
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metrics({
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        print("MAE :", mae)
        print("RMSE:", rmse)
        print("R2  :", r2)

        # =========================
        # Log model (WAJIB)
        # =========================
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_test.head(5)
        )

        print("Model berhasil disimpan sebagai artifact MLflow")

        # =========================
        # Simpan run_id SETELAH model berhasil di-log
        # =========================
        os.makedirs("MLProject", exist_ok=True)
        with open(RUN_ID_PATH, "w") as f:
            f.write(run.info.run_id)

        print(f"Run ID disimpan ke {RUN_ID_PATH}: {run.info.run_id}")


if __name__ == "__main__":
    run_model()

