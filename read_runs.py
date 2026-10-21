import mlflow  # Reuse MLflow to read the results

# Search for all runs in the experiment and sort by F1 score
# DESC = descending order (best score first)
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"]
)

# Select the columns to display
columns = [
    "tags.mlflow.runName",
    "params.n_estimators",
    "params.max_depth",
    "metrics.accuracy",
    "metrics.f1"
]

# Display the results without row numbers
print(runs[columns].to_string(index=False))

# The first row contains the best run
best = runs.iloc[0]

print()

# Display the best run's name and the first 8 characters of its ID
print(
    "Best run:",
    best["tags.mlflow.runName"],
    "| id:",
    best["run_id"][:8],
    "..."
)