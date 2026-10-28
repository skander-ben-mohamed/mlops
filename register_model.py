# ===== 1. Import the libraries =====
import mlflow
from mlflow import MlflowClient


# ===== 2. Connect to the same database as Lab 3 =====
# The mlflow.db file in the current folder stores all our runs.
mlflow.set_tracking_uri("sqlite:///mlflow.db")


# ===== 3. Find the best run =====
# Sort runs by F1-score in descending order.
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"]
)

# The first row contains the best run.
best = runs.iloc[0]
run_id = best["run_id"]

print(
    "Best run:",
    best["tags.mlflow.runName"],
    "| ID:",
    run_id[:8],
    "..."
)


# ===== 4. Register the best model =====
# "runs:/<run_id>/model" points to the model saved in the run.
# "churn-model" is the model's name in the MLflow Model Registry.
result = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name="churn-model"
)

print(
    "Model registered: churn-model | Version:",
    result.version
)


# ===== 5. Assign the 'champion' alias =====
# The alias identifies the current production candidate.
client = MlflowClient()

client.set_registered_model_alias(
    name="churn-model",
    alias="champion",
    version=result.version
)

print(
    "Alias 'champion' -> version",
    result.version
)