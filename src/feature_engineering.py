"""
feature_engineering.py
----------------------
Feature engineering pipeline for AtmosIQ.
"""

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from src.config import (
    CLEAN_DATA_PATH,
    ENGINEERED_DATA_PATH,
    TARGET_COLUMN,
    TARGET_ENCODER_PATH,
    SEASON_ENCODER_PATH,
)


class FeatureEngineer:

    def __init__(self):

        self.input_path = CLEAN_DATA_PATH
        self.output_path = ENGINEERED_DATA_PATH

        self.df = None

    # -------------------------------------------------------
    # Load Clean Dataset
    # -------------------------------------------------------

    def load_data(self):

        self.df = pd.read_csv(self.input_path)

        self.df["date"] = pd.to_datetime(self.df["date"])

        print(f"\n Loaded {len(self.df)} rows.")

    # -------------------------------------------------------
    # Date Features
    # -------------------------------------------------------

    def create_date_features(self):

        self.df["month"] = self.df["date"].dt.month
        self.df["day"] = self.df["date"].dt.day
        self.df["day_of_week"] = self.df["date"].dt.dayofweek
        self.df["day_of_year"] = self.df["date"].dt.dayofyear
        self.df["quarter"] = self.df["date"].dt.quarter
        self.df["week_of_year"] = (
            self.df["date"]
            .dt.isocalendar()
            .week
            .astype(int)
        )

    # -------------------------------------------------------
    # Cyclic Features
    # -------------------------------------------------------

    def create_cyclic_features(self):

        self.df["month_sin"] = np.sin(
            2 * np.pi * self.df["month"] / 12
        )

        self.df["month_cos"] = np.cos(
            2 * np.pi * self.df["month"] / 12
        )

    # -------------------------------------------------------
    # Weather Features
    # -------------------------------------------------------

    def create_weather_features(self):

        self.df["avg_temp"] = (
            self.df["temp_max"] +
            self.df["temp_min"]
        ) / 2

        self.df["temp_range"] = (
            self.df["temp_max"] -
            self.df["temp_min"]
        )

    # -------------------------------------------------------
    # Season Feature
    # -------------------------------------------------------

    def create_season(self):

        season_map = {
            12: "Winter",
            1: "Winter",
            2: "Winter",
            3: "Spring",
            4: "Spring",
            5: "Spring",
            6: "Summer",
            7: "Summer",
            8: "Summer",
            9: "Autumn",
            10: "Autumn",
            11: "Autumn",
        }

        self.df["season"] = self.df["month"].map(season_map)

    # -------------------------------------------------------
    # Encode Features
    # -------------------------------------------------------

    def encode_features(self):

        season_encoder = LabelEncoder()

        self.df["season"] = season_encoder.fit_transform(
            self.df["season"]
        )

        target_encoder = LabelEncoder()

        self.df[TARGET_COLUMN] = target_encoder.fit_transform(
            self.df[TARGET_COLUMN]
        )

        joblib.dump(
            season_encoder,
            SEASON_ENCODER_PATH
        )

        joblib.dump(
            target_encoder,
            TARGET_ENCODER_PATH
        )

    # -------------------------------------------------------
    # Drop Unused Columns
    # -------------------------------------------------------

    def drop_columns(self):

        self.df.drop(columns=["date"], inplace=True)

    # -------------------------------------------------------
    # Save Engineered Dataset
    # -------------------------------------------------------

    def save_dataset(self):

        self.df.to_csv(
            self.output_path,
            index=False
        )

        print(
            f"\n Engineered dataset saved to:\n{self.output_path}"
        )

    # -------------------------------------------------------
    # Pipeline
    # -------------------------------------------------------

    def run(self):

        print("\n========== FEATURE ENGINEERING ==========")

        self.load_data()

        self.create_date_features()

        self.create_cyclic_features()

        self.create_weather_features()

        self.create_season()

        self.encode_features()

        self.drop_columns()

        self.save_dataset()

        print("\n Feature Engineering Completed Successfully.")

        return self.df