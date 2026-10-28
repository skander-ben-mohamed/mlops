# ===== 1. Import the libraries =====
from typing import Literal

import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


# ===== 2. Create the application =====
app = FastAPI(title="Churn API")


# ===== 3. Load the champion model only once at startup =====
# Reloading the model for every request would make the API slow.
mlflow.set_tracking_uri("sqlite:///mlflow.db")

model = mlflow.sklearn.load_model("models:/churn-model@champion")


# ===== 4. Describe and validate the expected customer data =====
class Customer(BaseModel):
    age: int
    salaire: float
    abonnement: Literal["Basique", "Standard", "Premium"]
    anciennete: float
    consommation: float


# ===== 5. Route /health: check if the API is running =====
@app.get("/health")
def health():
    return {"status": "ok"}


# ===== 6. Route /predict: predict customer churn =====
@app.post("/predict")
def predict(customer: Customer):
    # a) Convert the customer into a one-row DataFrame.
    row = pd.DataFrame([customer.model_dump()])

    # b) Convert the subscription category into one-hot encoded columns.
    row = pd.get_dummies(row)

    # c) Keep the model's expected columns in the correct order.
    # Missing columns are filled with 0.
    row = row.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    # d) Predict churn: 1 = leaves, 0 = stays.
    result = int(model.predict(row)[0])

    return {
        "churn": result,
        "message": (
            "The customer will leave"
            if result == 1
            else "The customer stays"
        )
    }