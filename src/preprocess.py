"""
preprocess.py
-------------
Data preprocessing pipeline for AtmosIQ.
"""

import pandas as pd

from src.database import DatabaseManager
from src.config import PROCESSED_DATA_DIR


class DataPreprocessor:

    def __init__(self):
        self.db = DatabaseManager()
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------
    # Load Data
    # ------------------------------------

    def load_data(self):

        return self.db.load_data()

    # ------------------------------------
    # Inspect Dataset
    # ------------------------------------

    def inspect_data(self, df):

        print("\n========== DATASET INFO ==========\n")

        print(f"Shape : {df.shape}")

        print("\nData Types\n")
        print(df.dtypes)

        print("\nMissing Values\n")
        print(df.isnull().sum())

        print("\nDuplicate Rows :", df.duplicated().sum())

    # ------------------------------------
    # Clean Dataset
    # ------------------------------------

    def clean_data(self, df):

        df = df.copy()

        df.drop_duplicates(inplace=True)

        return df

    # ------------------------------------
    # Convert Date
    # ------------------------------------

    def convert_date(self, df):

        df["date"] = pd.to_datetime(df["date"])

        df.sort_values("date", inplace=True)

        df.reset_index(drop=True, inplace=True)

        return df

    # ------------------------------------
    # Save Dataset
    # ------------------------------------

    def save_processed_data(self, df):

        output_path = PROCESSED_DATA_DIR / "clean_weather.csv"

        df.to_csv(output_path, index=False)

        print(f"\n Clean dataset saved at:\n{output_path}")

    # ------------------------------------
    # Pipeline
    # ------------------------------------

    def run_pipeline(self):

        df = self.load_data()

        self.inspect_data(df)

        df = self.clean_data(df)

        df = self.convert_date(df)

        self.save_processed_data(df)

        return df