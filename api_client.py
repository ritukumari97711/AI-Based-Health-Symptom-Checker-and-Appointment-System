import requests

API_URL = "http://127.0.0.1:5000/api"


def predict(symptoms):
    response = requests.post(f"{API_URL}/predict", json={"symptoms": symptoms}, timeout=10)
    response.raise_for_status()
    return response.json()


def book_appointment(data):
    response = requests.post(f"{API_URL}/book", json=data, timeout=10)
    response.raise_for_status()
    return response.json()


def get_appointments():
    response = requests.get(f"{API_URL}/appointments", timeout=10)
    response.raise_for_status()
    return response.json()
