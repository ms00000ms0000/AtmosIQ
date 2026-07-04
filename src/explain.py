"""
explain.py
----------
Model explainability for AtmosIQ using SHAP.

Supports the tree-based classical models (Random Forest, Gradient
Boosting) with the fast TreeExplainer, and falls back to a
model-agnostic KernelExplainer for the TensorFlow Neural Network.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend, safe for Streamlit + scripts
import matplotlib.pyplot as plt
import shap

from src.config import FIGURE_DIR
from src.utils import (
    load_best_model,
    load_scaler,
    load_target_encoder,
    load_feature_frame,
    FEATURE_COLUMNS,
)

SHAP_SUMMARY_PATH = FIGURE_DIR / "shap_summary.png"
SHAP_BAR_PATH = FIGURE_DIR / "shap_feature_importance.png"

# KernelExplainer is O(n_samples * n_features) and slow, so we cap the
# background/explain sample sizes for the Neural Network fallback path.
MAX_BACKGROUND_SAMPLES = 100
MAX_EXPLAIN_SAMPLES = 200


class ModelExplainer:

    def __init__(self):
        self.model, self.model_name, self.is_nn = load_best_model()
        self.target_encoder = load_target_encoder()
        self.class_names = list(self.target_encoder.classes_)

        self.X = load_feature_frame()

        if self.is_nn:
            self.scaler = load_scaler()
        else:
            self.scaler = None

    # --------------------------------------------------
    # Build the right SHAP explainer for the winning model
    # --------------------------------------------------
    def _build_explainer(self):

        if not self.is_nn:
            # TreeExplainer is fast and exact for RandomForest / GradientBoosting.
            explainer = shap.TreeExplainer(self.model)
            X_sample = self.X.sample(
                min(len(self.X), 500), random_state=42
            )
            return explainer, X_sample

        # Neural Network: model-agnostic KernelExplainer on scaled data.
        background = shap.sample(
            self.X, min(len(self.X), MAX_BACKGROUND_SAMPLES), random_state=42
        )
        background_scaled = self.scaler.transform(background)

        def predict_fn(data):
            return self.model.predict(data, verbose=0)

        explainer = shap.KernelExplainer(predict_fn, background_scaled)

        X_sample = self.X.sample(
            min(len(self.X), MAX_EXPLAIN_SAMPLES), random_state=42
        )
        X_sample_scaled = self.scaler.transform(X_sample)

        return explainer, X_sample_scaled, X_sample

    # --------------------------------------------------
    # Compute SHAP values for the dataset (global explainability)
    # --------------------------------------------------
    def compute_shap_values(self):

        if not self.is_nn:
            explainer, X_sample = self._build_explainer()
            shap_values = explainer.shap_values(X_sample)
            return shap_values, X_sample

        explainer, X_sample_scaled, X_sample = self._build_explainer()
        shap_values = explainer.shap_values(X_sample_scaled, nsamples=100)
        return shap_values, X_sample

    # --------------------------------------------------
    # Global Summary Plot (which features matter most, overall)
    # --------------------------------------------------
    def plot_global_summary(self):

        shap_values, X_sample = self.compute_shap_values()

        plt.figure()

        # Multi-class models return a list of arrays (one per class);
        # shap's summary_plot handles that natively for bar plots.
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=FEATURE_COLUMNS,
            class_names=self.class_names,
            show=False,
            plot_type="bar",
        )

        plt.title(f"SHAP Feature Importance — {self.model_name}")
        plt.tight_layout()
        plt.savefig(SHAP_BAR_PATH, dpi=150)
        plt.close()

        print(f"\n SHAP feature importance plot saved to:\n{SHAP_BAR_PATH}")

        return SHAP_BAR_PATH

    # --------------------------------------------------
    # Explain a Single Prediction (local explainability)
    # --------------------------------------------------
    def explain_instance(self, feature_row):
        """
        feature_row : pd.DataFrame with one row, columns == FEATURE_COLUMNS
                       (unscaled — same format returned by
                       utils.build_feature_row).

        Returns a dict of {feature_name: shap_value} for the predicted class,
        sorted by absolute impact, for the single instance.
        """

        if not self.is_nn:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(feature_row)
            predicted_class = int(self.model.predict(feature_row)[0])

            # shap_values shape depends on sklearn/shap version:
            # list-of-arrays (per class) or a single 3D array.
            if isinstance(shap_values, list):
                values = shap_values[predicted_class][0]
            else:
                values = shap_values[0, :, predicted_class]

        else:
            background = shap.sample(
                self.X, min(len(self.X), MAX_BACKGROUND_SAMPLES), random_state=42
            )
            background_scaled = self.scaler.transform(background)

            def predict_fn(data):
                return self.model.predict(data, verbose=0)

            explainer = shap.KernelExplainer(predict_fn, background_scaled)

            scaled_row = self.scaler.transform(feature_row)
            probabilities = self.model.predict(scaled_row, verbose=0)[0]
            predicted_class = int(np.argmax(probabilities))

            shap_values = explainer.shap_values(scaled_row, nsamples=100)
            values = shap_values[predicted_class][0]

        impact = dict(zip(FEATURE_COLUMNS, values))

        return dict(
            sorted(impact.items(), key=lambda kv: abs(kv[1]), reverse=True)
        )


# --------------------------------------------------------
# Standalone Execution
# --------------------------------------------------------
if __name__ == "__main__":

    print("\n========== MODEL EXPLAINABILITY (SHAP) ==========\n")

    explainer = ModelExplainer()
    print(f"Explaining model: {explainer.model_name}")

    explainer.plot_global_summary()

    print("\nExplainability report generated successfully.")