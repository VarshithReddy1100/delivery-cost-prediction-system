import joblib
import pandas as pd


MODEL_PATH = "models/delivery_cost_model.pkl"


class DeliveryCostPredictor:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)

    def predict(self, input_data: dict):
        input_df = pd.DataFrame([input_data])
        prediction = self.model.predict(input_df)[0]
        return round(float(prediction), 2)


if __name__ == "__main__":
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

    result = predictor.predict(sample_input)

    print("Estimated Delivery Cost:", result)