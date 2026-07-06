"""
train.py
--------
Model Training Engine for AtmosIQ.

Methodology notes (why this file looks the way it does):

1. The dataset (now 10,000 rows, ~2000 per class after the synthetic
   expansion) is compared using Stratified 5-Fold Cross-Validation, with
   macro-F1 as the primary selection metric (treats every class equally
   regardless of size — important since even the expanded dataset can
   still have subtly harder classes like fog/drizzle).

2. SMOTE (Synthetic Minority Over-sampling) is applied INSIDE each CV
   fold via an imblearn Pipeline, never before the split, to avoid data
   leakage between folds/test set.

3. ALL SIX models are genuinely eligible to win, including the Neural
   Network: it's wrapped with scikeras.wrappers.KerasClassifier so it can
   run through the exact same StratifiedKFold/cross_validate machinery as
   the classical models, on equal footing. (Earlier versions of this file
   deliberately excluded the NN from competing since 5x-retraining a
   network per run was expensive on a small dataset where it clearly
   lost anyway — now that the dataset is bigger and the NN is more
   competitive, it's given a fair, identical evaluation process instead
   of a hardcoded disadvantage.)

4. To keep total runtime reasonable, the NN's CV phase uses a smaller
   epoch budget with early stopping (this is only for the fold-by-fold
   *comparison*, it doesn't need each fold to fully converge to give a
   fair relative ranking). The final NN that actually gets saved to disk
   is retrained separately with the full epoch budget for real
   convergence, exactly as before.

5. Persistence stays consistent with predict.py / explain.py / utils.py:
   classical winners are saved as a joblib-pickled
   SMOTE-in-pipeline + classifier, and the NN winner is saved as a plain
   .keras file (via model.save()) plus the fitted StandardScaler — no
   downstream files need to change regardless of which model wins.
"""

import time
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from scikeras.wrappers import KerasClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
)

from src.database import DatabaseManager

from src.config import (
    ENGINEERED_DATA_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
    VALIDATION_SIZE,
    RANDOM_STATE,
    MODEL_COMPARISON_PATH,
    BEST_MODEL_PATH,
    BEST_MODEL_NAME_PATH,
    TF_MODEL_PATH,
    CHECKPOINT_PATH,
    SCALER_PATH,
    TENSORBOARD_LOG_DIR,
)

# scikeras enables TensorFlow's deterministic-ops mode internally when the
# NN runs through cross_validate(). Once that's on, ANY subsequent Keras
# model built in the same process (e.g. the final, fully-converged NN fit
# after CV) needs an explicit global seed too, or ops like Dropout raise
# "Random ops require a seed to be set when determinism is enabled."
# Setting this once up front keeps both the CV phase and the final fit
# reproducible and crash-free.
tf.keras.utils.set_random_seed(RANDOM_STATE)

CV_FOLDS = 5

# The smallest class still needs SMOTE's k_neighbors to stay below its
# per-fold training count, so a conservative value is used everywhere
# instead of sklearn's default of 5.
SMOTE_K_NEIGHBORS = 3

# NN cross-validation uses a lighter epoch budget than the final,
# deployed fit — see module docstring point 4.
NN_CV_EPOCHS = 40
NN_CV_PATIENCE = 8
NN_FINAL_EPOCHS = 150
NN_FINAL_PATIENCE = 15


def make_smote_pipeline(model):
    """Wrap a classifier so SMOTE oversampling runs only on the training
    fold during .fit(), and is a no-op during .predict()/.predict_proba()."""
    return ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=SMOTE_K_NEIGHBORS)),
        ("clf", model),
    ])


def build_nn_for_scikeras(meta, hidden1=128, hidden2=64, dropout1=0.30, dropout2=0.20):
    """
    scikeras calls this with a `meta` dict describing the data (feature
    count, number of classes) once it knows the shape of whatever fold
    it's given — this is what lets the same architecture-builder work
    inside cross_validate() across differently-sized CV folds.
    """
    n_features_in = meta["n_features_in_"]
    n_classes = meta["n_classes_"]

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features_in,)),
        tf.keras.layers.Dense(hidden1, activation="relu"),
        tf.keras.layers.Dropout(dropout1),
        tf.keras.layers.Dense(hidden2, activation="relu"),
        tf.keras.layers.Dropout(dropout2),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


class ModelTrainer:

    def __init__(self):
        self.db = DatabaseManager()
        self.scaler = StandardScaler()
        self.results = []
        self.best_score = -1.0          # ranked by mean CV macro-F1
        self.best_model = None          # classical winner: fitted Pipeline
        self.best_model_name = None
        self.best_is_nn = False         # True if the NN wins

    # --------------------------------------------------
    # Load & Split Data
    # --------------------------------------------------
    def load_data(self):
        df = pd.read_csv(ENGINEERED_DATA_PATH)

        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]

        return train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

    # --------------------------------------------------
    # Neural Network Architecture (for the final, deployed fit)
    # --------------------------------------------------
    def build_neural_network(self, input_dim, output_dim):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.30),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Dense(output_dim, activation="softmax"),
        ])

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    # --------------------------------------------------
    # Cross-Validated Evaluation for Sklearn (tree) Models
    # --------------------------------------------------
    def evaluate_sklearn(self, pipeline, name, X_train, X_test, y_train, y_test):
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        cv_results = cross_validate(
            pipeline, X_train, y_train, cv=cv,
            scoring=["accuracy", "f1_macro"], n_jobs=-1,
        )

        cv_accuracy_mean = cv_results["test_accuracy"].mean()
        cv_accuracy_std = cv_results["test_accuracy"].std()
        cv_f1_macro_mean = cv_results["test_f1_macro"].mean()
        cv_f1_macro_std = cv_results["test_f1_macro"].std()

        start = time.time()
        pipeline.fit(X_train, y_train)
        training_time = round(time.time() - start, 3)

        predictions = pipeline.predict(X_test)
        self._record_result(
            name, cv_accuracy_mean, cv_accuracy_std, cv_f1_macro_mean,
            cv_f1_macro_std, y_test, predictions, training_time,
        )

        if cv_f1_macro_mean > self.best_score:
            self.best_score = cv_f1_macro_mean
            self.best_model = pipeline
            self.best_model_name = name
            self.best_is_nn = False

    # --------------------------------------------------
    # Shared metric bookkeeping
    # --------------------------------------------------
    def _record_result(self, name, cv_acc_mean, cv_acc_std, cv_f1_mean, cv_f1_std,
                        y_test, predictions, training_time):
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
        recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
        f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)
        f1_macro = f1_score(y_test, predictions, average="macro", zero_division=0)
        per_class_recall = recall_score(y_test, predictions, average=None, zero_division=0)

        self.results.append({
            "Model": name,
            "CV Accuracy (mean)": round(cv_acc_mean, 4),
            "CV Accuracy (std)": round(cv_acc_std, 4),
            "CV F1-Macro (mean)": round(cv_f1_mean, 4),
            "CV F1-Macro (std)": round(cv_f1_std, 4),
            "Test Accuracy": round(accuracy, 4),
            "Test Precision": round(precision, 4),
            "Test Recall": round(recall, 4),
            "Test F1 (weighted)": round(f1, 4),
            "Test F1 (macro)": round(f1_macro, 4),
            "Training Time (s)": training_time,
        })

        self.db.save_model_result(name, accuracy, precision, recall, f1)

        print(
            f"{name:<25} CV F1-Macro : {cv_f1_mean:.4f} (+/- {cv_f1_std:.4f})"
            f"   |   Test Accuracy : {accuracy:.4f}"
        )
        print(f"    Per-class recall (drizzle, fog, rain, snow, sun): "
              f"{np.round(per_class_recall, 3).tolist()}")

    # --------------------------------------------------
    # Random Forest / Extra Trees / Gradient Boosting / Hist GB
    # --------------------------------------------------
    def train_random_forest(self, X_train, X_test, y_train, y_test):
        model = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
        self.evaluate_sklearn(make_smote_pipeline(model), "Random Forest",
                               X_train, X_test, y_train, y_test)

    def train_extra_trees(self, X_train, X_test, y_train, y_test):
        model = ExtraTreesClassifier(n_estimators=400, random_state=RANDOM_STATE)
        self.evaluate_sklearn(make_smote_pipeline(model), "Extra Trees",
                               X_train, X_test, y_train, y_test)

    def train_gradient_boosting(self, X_train, X_test, y_train, y_test):
        model = GradientBoostingClassifier(random_state=RANDOM_STATE)
        self.evaluate_sklearn(make_smote_pipeline(model), "Gradient Boosting",
                               X_train, X_test, y_train, y_test)

    def train_hist_gradient_boosting(self, X_train, X_test, y_train, y_test):
        model = HistGradientBoostingClassifier(random_state=RANDOM_STATE, max_iter=300)
        self.evaluate_sklearn(make_smote_pipeline(model), "Hist Gradient Boosting",
                               X_train, X_test, y_train, y_test)

    # --------------------------------------------------
    # TensorFlow Neural Network — now a full CV competitor
    # --------------------------------------------------
    def train_tensorflow(self, X_train, X_test, y_train, y_test):
        """
        X_train/X_test here are the RAW (unscaled) feature frames — scaling
        now happens inside the pipeline (per-fold, like everything else)
        instead of once globally, so the NN's CV score isn't given an
        unfair leakage advantage from being scaled on the full training set.
        """

        cv_clf = KerasClassifier(
            model=build_nn_for_scikeras,
            epochs=NN_CV_EPOCHS,
            batch_size=32,
            verbose=0,
            random_state=RANDOM_STATE,
            callbacks=[EarlyStopping(monitor="loss", patience=NN_CV_PATIENCE,
                                      restore_best_weights=True)],
        )

        cv_pipeline = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=SMOTE_K_NEIGHBORS)),
            ("clf", cv_clf),
        ])

        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        print("  (Neural Network CV uses a lighter epoch budget per fold "
              "to keep total runtime reasonable — see docstring.)")

        cv_results = cross_validate(
            cv_pipeline, X_train, y_train, cv=cv,
            scoring=["accuracy", "f1_macro"], n_jobs=1,  # TF doesn't parallelize safely here
        )

        cv_accuracy_mean = cv_results["test_accuracy"].mean()
        cv_accuracy_std = cv_results["test_accuracy"].std()
        cv_f1_macro_mean = cv_results["test_f1_macro"].mean()
        cv_f1_macro_std = cv_results["test_f1_macro"].std()

        # ---- Final, fully-converged fit for the model that gets deployed ----
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=SMOTE_K_NEIGHBORS)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

        final_model = self.build_neural_network(
            X_train_res.shape[1], len(pd.unique(y_train_res))
        )

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=NN_FINAL_PATIENCE,
                           restore_best_weights=True),
            ModelCheckpoint(filepath=CHECKPOINT_PATH, monitor="val_accuracy",
                             save_best_only=True, verbose=0),
        ]
        try:
            import tensorboard  # noqa: F401
            callbacks.append(TensorBoard(log_dir=TENSORBOARD_LOG_DIR, histogram_freq=1))
        except ImportError:
            print("  [!] 'tensorboard' package not found — skipping TensorBoard logging.")

        start = time.time()
        final_model.fit(
            X_train_res, y_train_res,
            validation_split=VALIDATION_SIZE,
            epochs=NN_FINAL_EPOCHS, batch_size=32,
            callbacks=callbacks, verbose=0,
        )
        training_time = round(time.time() - start, 3)

        predictions = np.argmax(final_model.predict(X_test_scaled, verbose=0), axis=1)

        self._record_result(
            "Neural Network", cv_accuracy_mean, cv_accuracy_std,
            cv_f1_macro_mean, cv_f1_macro_std, y_test, predictions, training_time,
        )

        # Always save the fitted scaler + final model to disk — if the NN
        # doesn't win, these files are simply left unused. If it DOES win,
        # save_best_model() just needs to confirm the name; the artifacts
        # are already in place.
        joblib.dump(self.scaler, SCALER_PATH)
        final_model.save(TF_MODEL_PATH)

        if cv_f1_macro_mean > self.best_score:
            self.best_score = cv_f1_macro_mean
            self.best_model = None
            self.best_model_name = "Neural Network"
            self.best_is_nn = True

    # --------------------------------------------------
    # Train All Models
    # --------------------------------------------------
    def train(self):
        X_train, X_test, y_train, y_test = self.load_data()

        print("Class counts in training split BEFORE SMOTE:")
        print(y_train.value_counts().sort_index().to_dict())
        print()

        # Tree-based models: raw (unscaled) features, SMOTE applied inside
        # each pipeline/fold.
        self.train_random_forest(X_train, X_test, y_train, y_test)
        self.train_extra_trees(X_train, X_test, y_train, y_test)
        self.train_gradient_boosting(X_train, X_test, y_train, y_test)
        self.train_hist_gradient_boosting(X_train, X_test, y_train, y_test)

        # Neural network: scaling + SMOTE now happen inside its own
        # pipeline/fold too (see train_tensorflow).
        self.train_tensorflow(X_train, X_test, y_train, y_test)

    # --------------------------------------------------
    # Save Best Model
    # --------------------------------------------------
    def save_best_model(self):
        if self.best_is_nn:
            # The winning NN and its scaler were already saved to
            # TF_MODEL_PATH / SCALER_PATH inside train_tensorflow().
            with open(BEST_MODEL_NAME_PATH, "w") as file:
                file.write("Neural Network")
            print(
                f"\nBest model: Neural Network "
                f"(mean CV F1-Macro = {self.best_score:.4f}) — "
                f"already saved to {TF_MODEL_PATH.name}."
            )
        elif self.best_model is not None:
            joblib.dump(self.best_model, BEST_MODEL_PATH)
            with open(BEST_MODEL_NAME_PATH, "w") as file:
                file.write(self.best_model_name)
            print(
                f"\nBest model: {self.best_model_name} "
                f"(mean CV F1-Macro = {self.best_score:.4f}) — saved."
            )

    # --------------------------------------------------
    # Save Comparison Results
    # --------------------------------------------------
    def save_results(self):
        results = pd.DataFrame(self.results)
        results.sort_values(by="Test Accuracy", ascending=False, inplace=True)
        results.to_csv(MODEL_COMPARISON_PATH, index=False)

        print("\nModel comparison saved.")
        print("\n")
        print(results.to_string(index=False))

    # --------------------------------------------------
    # Main Pipeline
    # --------------------------------------------------
    def run(self):
        print("\n========== MODEL TRAINING ==========\n")
        print(f"Using {CV_FOLDS}-fold Stratified Cross-Validation for model selection.")
        print("Class imbalance fix: SMOTE oversampling (applied inside each fold).")
        print("All 6 models — including the Neural Network — compete on equal footing.")
        print("Selection metric: mean CV macro-F1 (robust to class imbalance).\n")

        self.train()
        self.save_best_model()
        self.save_results()

        print("\nTraining Completed Successfully.")


# --------------------------------------------------------
# Standalone Execution
# --------------------------------------------------------
if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run()