import mlflow
import mlflow.sklearn
import pandas as pd


# ===== 1. Find the best run =====
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"]
)

run_id = runs.iloc[0]["run_id"]


# ===== 2. Reload the best model =====
# "runs:/<id>/model" points to the model saved in this run
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

print("Model reloaded:", type(model).__name__)
print("Number of expected variables:", model.n_features_in_)


# ===== 3. Create a new customer =====
# Standard subscription: Premium = False
customer = pd.DataFrame([{
    "age": 35,
    "salaire": 2500,
    "anciennete": 1,
    "consommation": 10,
    "abonnement_Premium": False,
    "abonnement_Standard": True
}])


# ===== 4. Keep the expected columns in the correct order =====
# The selected model may use a different set of features
customer = customer[list(model.feature_names_in_)]


# ===== 5. Predict customer churn =====
# 1 = the customer will leave, 0 = the customer will stay
prediction = model.predict(customer)[0]

print(
    "Will the customer leave? ->",
    "YES" if prediction == 1 else "NO"
)