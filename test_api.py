import requests  # Send HTTP requests from Python


# ===== 1. Define the API URL =====
URL = "http://localhost:8000"


# ===== 2. Check if the API is alive =====
response = requests.get(URL + "/health")

print("GET /health ->", response.json())


# ===== 3. Test customer 1: low usage and low seniority =====
customer = {
    "age": 35,
    "salaire": 2500,
    "abonnement": "Standard",
    "anciennete": 1,
    "consommation": 10
}

response = requests.post(
    URL + "/predict",
    json=customer
)

print(
    "POST /predict (customer 1) ->",
    response.status_code,
    response.json()
)


# ===== 4. Test customer 2: high usage and high seniority =====
customer2 = {
    "age": 50,
    "salaire": 4000,
    "abonnement": "Premium",
    "anciennete": 10,
    "consommation": 80
}

response = requests.post(
    URL + "/predict",
    json=customer2
)

print(
    "POST /predict (customer 2) ->",
    response.status_code,
    response.json()
)


# ===== 5. Test input validation with invalid data =====
# Copy customer 1 and change age to an invalid string.
bad = dict(customer, age="abc")

response = requests.post(
    URL + "/predict",
    json=bad
)

print(
    "POST /predict (age='abc') ->",
    response.status_code
)

print("Message:", response.json()["detail"][0]["msg"])