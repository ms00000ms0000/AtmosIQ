from src.database import DatabaseManager
from src.preprocess import DataPreprocessor
from src.eda import EDA
from src.feature_engineering import FeatureEngineer


def main():

    print("\n" + "=" * 60)
    print(" AtmosIQ - Weather Intelligence Platform ")
    print("=" * 60)

    # ---------------------------------------
    # Database Setup
    # ---------------------------------------

    db = DatabaseManager()

    db.create_tables()

    db.import_csv()

    # ---------------------------------------
    # Data Preprocessing
    # ---------------------------------------

    preprocessor = DataPreprocessor()

    clean_df = preprocessor.run_pipeline()

    # ---------------------------------------
    # Exploratory Data Analysis
    # ---------------------------------------

    eda = EDA(clean_df)

    eda.run()

    # ---------------------------------------
    # Feature Engineering
    # ---------------------------------------

    feature_engineer = FeatureEngineer()

    engineered_df = feature_engineer.run()

    # ---------------------------------------
    # Summary
    # ---------------------------------------

    print("\n" + "=" * 60)
    print(" PIPELINE COMPLETED SUCCESSFULLY ")
    print("=" * 60)

    print(f"\nFinal Dataset Shape : {engineered_df.shape}")

    print("\nEngineered Dataset Preview:\n")

    print(engineered_df.head())

    db.close()


if __name__ == "__main__":
    main()