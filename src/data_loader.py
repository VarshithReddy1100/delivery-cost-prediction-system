import pandas as pd
import os


DATA_PATH = "data/delivery_data.csv"


def load_dataset():

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("\n" + "=" * 60)
    print("DATASET LOADED SUCCESSFULLY")
    print("=" * 60)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    print("\nFirst 5 Rows:")
    print(df.head())

    return df


if __name__ == "__main__":
    load_dataset()