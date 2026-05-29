import joblib

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

from src.data_loader import load_dataset
from src.preprocessing import prepare_data


def tune_random_forest():

    df = load_dataset()

    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(random_state=42))
        ]
    )

    param_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [5, 10, 15, 20, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4]
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=10,
        cv=3,
        scoring="r2",
        random_state=42,
        n_jobs=-1
    )

    print("Starting Hyperparameter Tuning...")

    search.fit(X_train, y_train)

    print("\nBest Parameters:")
    print(search.best_params_)

    print("\nBest CV Score:")
    print(search.best_score_)

    joblib.dump(
        search.best_estimator_,
        "models/tuned_delivery_cost_model.pkl"
    )

    print(
        "\nTuned model saved to models/tuned_delivery_cost_model.pkl"
    )


if __name__ == "__main__":
    tune_random_forest()