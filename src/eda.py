import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_dataset

os.makedirs("reports/figures", exist_ok=True)

df = load_dataset()

print("\nDataset Statistics:")
print(df.describe())

# -------------------------
# Missing Values Heatmap
# -------------------------
plt.figure(figsize=(10, 5))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Values Heatmap")
plt.tight_layout()
plt.savefig("reports/figures/null_heatmap.png")
plt.close()

# -------------------------
# Correlation Matrix
# -------------------------
numeric_df = df.select_dtypes(include=["int64", "float64"])

plt.figure(figsize=(10, 8))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("reports/figures/correlation_matrix.png")
plt.close()

# -------------------------
# Target Distribution
# -------------------------
plt.figure(figsize=(8, 5))
sns.histplot(df["delivery_cost"], bins=30, kde=True)
plt.title("Delivery Cost Distribution")
plt.tight_layout()
plt.savefig("reports/figures/target_distribution.png")
plt.close()

# -------------------------
# Delivery Partner Analysis
# -------------------------
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="delivery_partner")
plt.xticks(rotation=45)
plt.title("Delivery Partner Distribution")
plt.tight_layout()
plt.savefig("reports/figures/delivery_partner_distribution.png")
plt.close()

# -------------------------
# Weather Condition Analysis
# -------------------------
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="weather_condition")
plt.title("Weather Distribution")
plt.tight_layout()
plt.savefig("reports/figures/weather_distribution.png")
plt.close()

# -------------------------
# Vehicle Type Analysis
# -------------------------
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="vehicle_type")
plt.title("Vehicle Type Distribution")
plt.tight_layout()
plt.savefig("reports/figures/vehicle_distribution.png")
plt.close()

print("\nEDA Completed Successfully")
print("Figures saved in reports/figures/")