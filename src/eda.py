"""
eda.py
------
Exploratory Data Analysis for AtmosIQ.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import REPORT_DIR


class EDA:

    def __init__(self, df):

        self.df = df.copy()

        self.figure_dir = REPORT_DIR / "figures"

        self.figure_dir.mkdir(parents=True, exist_ok=True)

        sns.set_style("whitegrid")

    # ---------------------------------
    # Dataset Summary
    # ---------------------------------

    def dataset_summary(self):

        print("\n========== DATASET SUMMARY ==========\n")

        print(self.df.describe(include="all"))

    # ---------------------------------
    # Class Distribution
    # ---------------------------------

    def class_distribution(self):

        plt.figure(figsize=(8,5))

        sns.countplot(data=self.df, x="weather")

        plt.title("Weather Class Distribution")

        plt.tight_layout()

        plt.savefig(self.figure_dir / "class_distribution.png")

        plt.close()

    # ---------------------------------
    # Correlation Heatmap
    # ---------------------------------

    def correlation_heatmap(self):

        plt.figure(figsize=(8,6))

        corr = self.df.select_dtypes(include="number").corr()

        sns.heatmap(
            corr,
            annot=True,
            cmap="Blues",
            fmt=".2f"
        )

        plt.title("Correlation Heatmap")

        plt.tight_layout()

        plt.savefig(self.figure_dir / "correlation_heatmap.png")

        plt.close()

    # ---------------------------------
    # Temperature Distribution
    # ---------------------------------

    def temperature_distribution(self):

        plt.figure(figsize=(8,5))

        sns.histplot(
            self.df["temp_max"],
            bins=30,
            kde=True
        )

        plt.title("Maximum Temperature Distribution")

        plt.tight_layout()

        plt.savefig(self.figure_dir / "temperature_distribution.png")

        plt.close()

    # ---------------------------------
    # Wind Distribution
    # ---------------------------------

    def wind_distribution(self):

        plt.figure(figsize=(8,5))

        sns.histplot(
            self.df["wind"],
            bins=30,
            kde=True
        )

        plt.title("Wind Distribution")

        plt.tight_layout()

        plt.savefig(self.figure_dir / "wind_distribution.png")

        plt.close()

    # ---------------------------------
    # Monthly Trend
    # ---------------------------------

    def monthly_temperature(self):

        temp = self.df.copy()

        temp["month"] = temp["date"].dt.month

        monthly = temp.groupby("month")["temp_max"].mean()

        plt.figure(figsize=(10,5))

        monthly.plot(marker="o")

        plt.title("Average Monthly Maximum Temperature")

        plt.xlabel("Month")

        plt.ylabel("Temperature")

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(self.figure_dir / "monthly_temperature.png")

        plt.close()

    # ---------------------------------
    # Run Complete EDA
    # ---------------------------------

    def run(self):

        self.dataset_summary()

        self.class_distribution()

        self.correlation_heatmap()

        self.temperature_distribution()

        self.wind_distribution()

        self.monthly_temperature()

        print("\n EDA Completed")