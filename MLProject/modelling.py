import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn

path = "dataset_mesin_membangun_sistem_machine_learning_preprocessing.csv"
experiment_name = "Student Performance - Modelling"

mlflow.autolog()

def run_model():
    mlflow.set_experiment(experiment_name)

    df = pd.read_csv(path)

    X = df.drop(columns=["FinalGrade"])
    y = df["FinalGrade"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run() as run:
        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)

        print(f"Run ID disimpan: {run.info.run_id}")

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

        model.fit(X_train, y_train)

        print("Training selesai")

if __name__ == "__main__":
    run_model()
