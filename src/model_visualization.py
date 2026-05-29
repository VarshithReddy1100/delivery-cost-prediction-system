import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_dataset
from src.preprocessing import prepare_data

# Load data
df = load_dataset()

X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)

# Load saved model
model = joblib.load("models/delivery_cost_model.pkl")

# Predictions
y_pred = model.predict(X_test)

# --------------------------
# Actual vs Predicted
# --------------------------

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Cost")
plt.ylabel("Predicted Cost")
plt.title("Actual vs Predicted")
plt.tight_layout()
plt.savefig("reports/figures/actual_vs_predicted.png")
plt.close()

# --------------------------
# Residual Plot
# --------------------------

residuals = y_test - y_pred

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig("reports/figures/residual_plot.png")
plt.close()

# --------------------------
# Model Comparison
# --------------------------

with open("models/model_metrics.json") as f:
    metrics = json.load(f)

models = []
rmse_scores = []

for model_name, values in metrics["all_model_metrics"].items():
    models.append(model_name)
    rmse_scores.append(values["RMSE"])

plt.figure(figsize=(8, 5))
sns.barplot(x=models, y=rmse_scores)
plt.title("Model Comparison (RMSE)")
plt.ylabel("RMSE")
plt.tight_layout()
plt.savefig("reports/figures/model_comparison.png")
plt.close()

print("Evaluation Visualizations Generated Successfully")