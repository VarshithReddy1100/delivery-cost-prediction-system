import os
import json
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from src.data_loader import load_dataset
from src.preprocessing import prepare_data
from src.evaluate_model import evaluate_model


def train_models():
    df = load_dataset()

    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)

    models = {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42
        )
    }

    results = {}
    best_model = None
    best_model_name = None
    best_r2 = -999

    for model_name, model in models.items():
        print("\n" + "=" * 50)
        print(f"Training Model: {model_name}")
        print("=" * 50)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(
            pipeline,
            X_test,
            y_test
        )

        results[model_name] = metrics

        print(f"Results for {model_name}:")
        print(metrics)

        if metrics["R2_Score"] > best_r2:
            best_r2 = metrics["R2_Score"]
            best_model = pipeline
            best_model_name = model_name

    os.makedirs("models", exist_ok=True)

    joblib.dump(
        best_model,
        "models/delivery_cost_model.pkl"
    )

    final_results = {
        "best_model": best_model_name,
        "best_r2_score": best_r2,
        "all_model_metrics": results
    }

    with open("models/model_metrics.json", "w") as file:
        json.dump(final_results, file, indent=4)

    print("\n" + "=" * 50)
    print("TRAINING COMPLETED")
    print("=" * 50)
    print("Best Model:", best_model_name)
    print("Best R2 Score:", best_r2)
    print("Model saved at: models/delivery_cost_model.pkl")
    print("Metrics saved at: models/model_metrics.json")


if __name__ == "__main__":
    train_models()