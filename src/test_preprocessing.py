from src.data_loader import load_dataset
from src.preprocessing import prepare_data

df = load_dataset()

prepare_data(df)