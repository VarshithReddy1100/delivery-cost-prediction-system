from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import DeliveryCostPredictor


app = FastAPI(
    title="Real-Time Delivery Cost Estimation API",
    description="ML API for predicting delivery cost using logistics dataset",
    version="1.0.0"
)

predictor = DeliveryCostPredictor()


class DeliveryInput(BaseModel):
    delivery_partner: str
    package_type: str
    vehicle_type: str
    delivery_mode: str
    region: str
    weather_condition: str
    distance_km: float
    package_weight_kg: float
    delivery_time_hours: float
    expected_time_hours: float
    delayed: str
    delivery_status: str
    delivery_rating: int


@app.get("/")
def home():
    return {
        "message": "Real-Time Delivery Cost Estimation API is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "loaded"
    }


@app.post("/predict")
def predict_delivery_cost(data: DeliveryInput):
    input_data = data.model_dump()
    prediction = predictor.predict(input_data)

    return {
        "estimated_delivery_cost": prediction,
        "currency": "INR",
        "status": "success"
    }