"""
predict.py
----------
Prediction engine for AtmosIQ.

Loads whichever model won training (Random Forest, Gradient Boosting, or
the TensorFlow Neural Network), builds engineered features from raw user
input, returns a predicted weather label with a confidence score, and logs
every prediction to the SQL prediction_history table.
"""

import numpy as np

from src.database import DatabaseManager

from src.utils import (
    load_best_model,
    load_scaler,
    load_season_encoder,
    load_target_encoder,
    build_feature_row,
    decode_prediction,
    FEATURE_COLUMNS,
)


class WeatherPredictor:

    def __init__(self, log_to_db: bool = True):
        """
        Parameters
        ----------
        log_to_db : bool
            If True, every call to predict() is written to the
            prediction_history table. Set False for bulk/batch use where
            you don't want to flood the history table.
        """

        self.model, self.model_name, self.is_nn = load_best_model()
        self.scaler = load_scaler()
        self.season_encoder = load_season_encoder()
        self.target_encoder = load_target_encoder()

        self.log_to_db = log_to_db
        self.db = DatabaseManager() if log_to_db else None

    # --------------------------------------------------
    # Build model-ready features from raw input
    # --------------------------------------------------
    def _prepare_features(self, date, precipitation, temp_max, temp_min, wind):
        features = build_feature_row(
            date=date,
            precipitation=precipitation,
            temp_max=temp_max,
            temp_min=temp_min,
            wind=wind,
            season_encoder=self.season_encoder,
        )

        # Guard against silent column-order drift.
        features = features[FEATURE_COLUMNS]

        if self.is_nn:
            features = self.scaler.transform(features)

        return features

    # --------------------------------------------------
    # Predict Probabilities
    # --------------------------------------------------
    def _predict_proba(self, features):
        if self.is_nn:
            return self.model.predict(features, verbose=0)[0]

        return self.model.predict_proba(features)[0]

    # --------------------------------------------------
    # Public Predict API
    # --------------------------------------------------
    def predict(
        self,
        date,
        precipitation: float,
        temp_max: float,
        temp_min: float,
        wind: float,
    ) -> dict:
        """
        Run a single prediction.

        Returns
        -------
        dict with keys:
            label        : predicted weather class (e.g. "rain")
            confidence   : probability of the predicted class (0-1)
            probabilities: dict of {class_label: probability} for all classes
            model_used   : name of the model that produced this prediction
        """

        features = self._prepare_features(
            date, precipitation, temp_max, temp_min, wind
        )

        probabilities = self._predict_proba(features)

        predicted_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_index])

        label = decode_prediction(predicted_index, self.target_encoder)

        class_probabilities = {
            decode_prediction(i, self.target_encoder): float(prob)
            for i, prob in enumerate(probabilities)
        }

        if self.log_to_db and self.db is not None:
            self.db.save_prediction(label, confidence)

        return {
            "label": label,
            "confidence": confidence,
            "probabilities": class_probabilities,
            "model_used": self.model_name,
        }

    def close(self):
        if self.db is not None:
            self.db.close()


# --------------------------------------------------------
# Standalone / CLI Usage
# --------------------------------------------------------
if __name__ == "__main__":

    predictor = WeatherPredictor()

    result = predictor.predict(
        date="2026-07-04",
        precipitation=2.5,
        temp_max=28.0,
        temp_min=18.0,
        wind=3.2,
    )

    print("\n========== PREDICTION ==========\n")
    print(f"Model Used   : {result['model_used']}")
    print(f"Prediction   : {result['label']}")
    print(f"Confidence   : {result['confidence']:.2%}")
    print("\nClass Probabilities:")
    for label, prob in sorted(
        result["probabilities"].items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {label:<10}: {prob:.2%}")

    predictor.close()