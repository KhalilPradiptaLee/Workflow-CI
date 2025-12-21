import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

path = "dataset_mesin_membangun_sistem_machine_learning_preprocessing.csv"
experiment_name = "Student Performance - Modelling"

def run_model():
    mlflow.set_experiment(experiment_name)

    df = pd.read_csv(path)

    X = df.drop(columns=["FinalGrade"])
    y = df["FinalGrade"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # simpan run_id
        with open("run_id.txt", "w") as f:
            f.write(run_id)

        print(f"Run ID disimpan: {run_id}")

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mlflow.log_metrics({
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2": r2_score(y_test, y_pred)
        })

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_test.head(5)
        )

        print("Model berhasil disimpan sebagai artifact MLflow")

if __name__ == "__main__":
    run_model()
