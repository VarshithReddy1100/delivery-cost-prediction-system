import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import DeliveryCostPredictor


def test_prediction_output():
    predictor = DeliveryCostPredictor()

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

    prediction = predictor.predict(sample_input)

    assert isinstance(prediction, float)
    assert prediction > 0