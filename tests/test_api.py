import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from backend.app import app


client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_endpoint():
    sample_input = {
        "delivery_partner": "delhivery",
        "package_type": "electronics",
        "vehicle_type": "bike",
        "delivery_mode": "standard",
        "region": "south",
        "weather_condition": "clear",
        "distance_km": 25.5,
        "package_weight_kg": 8.2,
        "delivery_time_hours": 5.0,
        "expected_time_hours": 4.5,
        "delayed": "no",
        "delivery_status": "delivered",
        "delivery_rating": 4
    }

    response = client.post("/predict", json=sample_input)

    assert response.status_code == 200

    result = response.json()

    assert "estimated_delivery_cost" in result
    assert result["estimated_delivery_cost"] > 0
    assert result["currency"] == "INR"
    assert result["status"] == "success"