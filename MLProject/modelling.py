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
path = "dataset_mesin_membangun_sistem_machine_learning_preprocessing.csv"
experiment_name = "Student Performance - Modelling"


def run_model():

    # =========================
    # MLflow setup (local)
    # =========================
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(experiment_name)

    # Autolog (WAJIB untuk Basic)
    mlflow.sklearn.autolog()
    print("Training dimulai...")

    # =========================
    # Load dataset
    # =========================
    try:
        df = pd.read_csv(path)
        print("Dataset berhasil diload")
    except FileNotFoundError:
        print(f"Dataset tidak ditemukan di {path}")
        return

    # =========================
    # Feature & target
    # =========================
    x = df.drop(columns=["FinalGrade"])
    y = df["FinalGrade"]

    # =========================
    # Train-test split
    # =========================
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # =========================
    # Training
    # =========================
    with mlflow.start_run(run_name="RandomForest-Baseline"):
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

        model.fit(x_train, y_train)

        # =========================
        # Evaluation
        # =========================
        y_pred = model.predict(x_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print("MAE :", mae)
        print("RMSE:", rmse)
        print("R2  :", r2)


if __name__ == "__main__":
    run_model()
