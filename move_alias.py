import mlflow
from mlflow import MlflowClient


# ===== 1. Connect to the MLflow database =====
mlflow.set_tracking_uri("sqlite:///mlflow.db")

client = MlflowClient()


# ===== 2. Find the run "trial-2" =====
# trial-2 is the weakest model from Lab 3.
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    filter_string="tags.mlflow.runName = 'trial-2'"
)

run_id = runs.iloc[0]["run_id"]


# ===== 3. Register the model =====
# This creates a new version of the existing "churn-model".
v = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name="churn-model"
)

print("New version created:", v.version)


# ===== 4. Get the current champion =====
current = client.get_model_version_by_alias(
    name="churn-model",
    alias="champion"
)

print("Current champion: version", current.version)


# ===== 5. Move the champion alias to the new version =====
client.set_registered_model_alias(
    name="churn-model",
    alias="champion",
    version=v.version
)

print(
    "After the move: champion -> version",
    client.get_model_version_by_alias(
        "churn-model",
        "champion"
    ).version
)


# ===== 6. Restore the original champion =====
client.set_registered_model_alias(
    name="churn-model",
    alias="champion",
    version=current.version
)

print(
    "After going back: champion -> version",
    client.get_model_version_by_alias(
        "churn-model",
        "champion"
    ).version
)