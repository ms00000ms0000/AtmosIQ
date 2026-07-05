"""
streamlit_app.py
----------------
AtmosIQ - Weather Intelligence Platform
Modern Streamlit dashboard: EDA, model comparison, live prediction with
confidence scores, SHAP explainability, and SQL-backed prediction history.
"""

import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    FIGURE_DIR,
    MODEL_COMPARISON_PATH,
)
from src.database import DatabaseManager
from src.predict import WeatherPredictor
from src.explain import ModelExplainer, SHAP_BAR_PATH
from src.utils import (
    load_feature_frame,
    load_engineered_data,
    load_target_encoder,
    build_feature_row,
    weather_icon,
)

# ======================================================
# Page Configuration
# ======================================================

st.set_page_config(
    page_title="AtmosIQ | Weather Intelligence",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
.main-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #2563eb, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.subtitle {
    color: #64748b;
    font-size: 1.05rem;
    margin-top: 0.2rem;
}
.metric-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    text-align: center;
}
.result-card {
    background: linear-gradient(135deg, #eff6ff, #ecfeff);
    border: 1px solid #bfdbfe;
    border-radius: 18px;
    padding: 1.8rem;
    text-align: center;
}
.result-emoji {
    font-size: 4rem;
}
.result-label {
    font-size: 2rem;
    font-weight: 800;
    text-transform: capitalize;
    color: #1e293b;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ======================================================
# Cached Resource / Data Loaders
# ======================================================

@st.cache_resource
def get_predictor():
    return WeatherPredictor()


@st.cache_resource
def get_explainer():
    return ModelExplainer()


@st.cache_data
def get_engineered_data():
    return load_engineered_data()


@st.cache_data
def get_model_comparison():
    return pd.read_csv(MODEL_COMPARISON_PATH)


@st.cache_resource
def get_target_encoder():
    return load_target_encoder()


def get_db():
    # Not cached: SQLite writes happen from multiple parts of the app, so a
    # short-lived connection per call keeps things simple and consistent.
    return DatabaseManager()


@st.cache_resource
def init_database():
    """
    Ensure prediction_history / model_results tables exist.

    On a fresh deployment (e.g. Streamlit Community Cloud), main.py never
    runs, so nothing has created these tables yet. create_tables() is
    idempotent (CREATE TABLE IF NOT EXISTS), so it's safe to call on every
    app startup. st.cache_resource makes sure it only actually runs once
    per app session instead of on every rerun.
    """
    db = DatabaseManager()
    db.create_tables()
    db.close()
    return True


# Run once, before any page tries to read from the database.
init_database()


# ======================================================
# Sidebar Navigation
# ======================================================

st.sidebar.markdown("## 🌤️ AtmosIQ")
st.sidebar.caption("Weather Intelligence Platform")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Overview",
        "📊 EDA Dashboard",
        "🤖 Model Comparison",
        "🔮 Make a Prediction",
        "🧠 Explainability",
        "🗂️ Prediction History",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Built with Streamlit · Scikit-learn · TensorFlow · SHAP")


# ======================================================
# PAGE: Overview
# ======================================================

if page == "🏠 Overview":

    st.markdown('<p class="main-title">AtmosIQ</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">An end-to-end weather intelligence platform — '
        "from raw data to explainable ML predictions.</p>",
        unsafe_allow_html=True,
    )
    st.write("")

    df = get_engineered_data()
    comparison = get_model_comparison()
    db = get_db()
    history = db.get_prediction_history()
    db.close()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f'<div class="metric-card"><h2>{len(df)}</h2>'
            "<p>Historical Records</p></div>",
            unsafe_allow_html=True,
        )
    with col2:
        # The Neural Network has no CV score (NaN) since it's a documented
        # baseline, not a CV-eligible candidate — na_position="last" keeps
        # it out of contention here, matching train.py's actual selection.
        best_row = comparison.sort_values(
            "CV F1-Macro (mean)", ascending=False, na_position="last"
        ).iloc[0]
        st.markdown(
            f'<div class="metric-card"><h2>{best_row["Test Accuracy"]:.1%}</h2>'
            f'<p>Best Model: {best_row["Model"]}</p></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="metric-card"><h2>{len(comparison)}</h2>'
            "<p>Models Trained</p></div>",
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f'<div class="metric-card"><h2>{len(history)}</h2>'
            "<p>Predictions Logged</p></div>",
            unsafe_allow_html=True,
        )

    st.write("")
    st.subheader("What AtmosIQ Does")
    st.markdown(
        """
- **SQL-backed pipeline** — raw data is imported into SQLite and every model
  run / prediction is logged for full traceability.
- **Advanced feature engineering** — cyclic month encodings, seasonal
  buckets, temperature range & average, calendar features.
- **Five trained models** — Random Forest, Gradient Boosting, Hist Gradient Boosting, Extra Trees and a
  TensorFlow Neural Network, auto-compared and the best one promoted.
- **Explainable AI** — SHAP tells you *why* a prediction was made, not just
  what it was.
- **Live predictions with confidence scores**, all queryable from history.
        """
    )

    st.info(
        "Use the sidebar to explore the data, compare models, make a live "
        "prediction, or inspect what drives the model's decisions."
    )


# ======================================================
# PAGE: EDA Dashboard
# ======================================================

elif page == "📊 EDA Dashboard":

    st.header("📊 Exploratory Data Analysis")
    df = get_engineered_data()
    target_encoder = get_target_encoder()

    df_display = df.copy()
    df_display["weather"] = target_encoder.inverse_transform(df["weather"])

    tab1, tab2, tab3 = st.tabs(
        ["Class & Correlation", "Distributions", "Seasonal Trends"]
    )

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            counts = df_display["weather"].value_counts().reset_index()
            counts.columns = ["weather", "count"]
            fig = px.bar(
                counts,
                x="weather",
                y="count",
                color="weather",
                title="Weather Class Distribution",
                text="count",
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width="stretch")

        with col2:
            numeric_cols = [
                "precipitation", "temp_max", "temp_min", "wind",
                "avg_temp", "temp_range",
            ]
            corr = df[numeric_cols].corr()
            fig = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="Blues",
                title="Correlation Heatmap",
            )
            st.plotly_chart(fig, width="stretch")

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(
                df_display,
                x="temp_max",
                nbins=30,
                marginal="box",
                title="Maximum Temperature Distribution",
                color_discrete_sequence=["#2563eb"],
            )
            st.plotly_chart(fig, width="stretch")

        with col2:
            fig = px.histogram(
                df_display,
                x="wind",
                nbins=30,
                marginal="box",
                title="Wind Speed Distribution",
                color_discrete_sequence=["#06b6d4"],
            )
            st.plotly_chart(fig, width="stretch")

        fig = px.box(
            df_display,
            x="weather",
            y="precipitation",
            color="weather",
            title="Precipitation by Weather Type",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")

    with tab3:
        monthly = df_display.groupby("month")["temp_max"].mean().reset_index()
        fig = px.line(
            monthly,
            x="month",
            y="temp_max",
            markers=True,
            title="Average Monthly Maximum Temperature",
        )
        st.plotly_chart(fig, width="stretch")

        season_counts = (
            df_display.groupby(["season", "weather"])
            .size()
            .reset_index(name="count")
        )
        fig = px.bar(
            season_counts,
            x="season",
            y="count",
            color="weather",
            title="Weather Type by Season (encoded season index)",
            barmode="stack",
        )
        st.plotly_chart(fig, width="stretch")

    with st.expander("View static EDA report figures (saved during pipeline run)"):
        figs = sorted(FIGURE_DIR.glob("*.png"))
        cols = st.columns(3)
        for i, fig_path in enumerate(figs):
            if "shap" in fig_path.name:
                continue
            with cols[i % 3]:
                st.image(str(fig_path), caption=fig_path.stem.replace("_", " ").title())


# ======================================================
# PAGE: Model Comparison
# ======================================================

elif page == "🤖 Model Comparison":

    st.header("🤖 Model Comparison")
    st.caption(
        "Models are ranked by mean 5-fold cross-validated macro-F1 — this "
        "is more robust than a single train/test split on a dataset this "
        "small, and macro-F1 (vs. plain accuracy) prevents the ranking from "
        "hiding poor performance on minority weather classes like snow."
    )

    raw_comparison = get_model_comparison()

    # Neural Network has no CV score (NaN) — it's a documented baseline,
    # not a candidate for the "best model" title. na_position keeps it at
    # the bottom of the CV-ranked table without erroring on the NaN.
    comparison = raw_comparison.sort_values(
        "CV F1-Macro (mean)", ascending=False, na_position="last"
    )

    st.dataframe(
        comparison.style.format(
            {
                "CV Accuracy (mean)": "{:.2%}",
                "CV Accuracy (std)": "±{:.3f}",
                "CV F1-Macro (mean)": "{:.2%}",
                "CV F1-Macro (std)": "±{:.3f}",
                "Test Accuracy": "{:.2%}",
                "Test Precision": "{:.2%}",
                "Test Recall": "{:.2%}",
                "Test F1 (weighted)": "{:.2%}",
                "Test F1 (macro)": "{:.2%}",
                "Training Time (s)": "{:.2f}s",
            },
            na_rep="—",
        ).background_gradient(cmap="Blues", subset=["CV F1-Macro (mean)"]),
        width="stretch",
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            comparison,
            x="Model",
            y=["Test Accuracy", "Test F1 (weighted)", "Test F1 (macro)"],
            barmode="group",
            title="Test-Set Metrics Across Models",
        )
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, width="stretch")

    with col2:
        cv_only = comparison.dropna(subset=["CV F1-Macro (mean)"])
        fig = px.bar(
            cv_only,
            x="Model",
            y="CV F1-Macro (mean)",
            error_y="CV F1-Macro (std)",
            title="5-Fold CV Macro-F1 (± std) — Selection Metric",
            color="Model",
        )
        fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(fig, width="stretch")

    best_row = comparison.iloc[0]
    st.success(
        f"🏆 **{best_row['Model']}** is the current champion with "
        f"**{best_row['CV F1-Macro (mean)']:.2%}** mean CV macro-F1 "
        f"(test accuracy: {best_row['Test Accuracy']:.2%}) — this is the "
        "model used for live predictions."
    )


# ======================================================
# PAGE: Make a Prediction
# ======================================================

elif page == "🔮 Make a Prediction":

    st.header("🔮 Make a Prediction")
    st.caption(
        "Enter tomorrow's forecast inputs and AtmosIQ will predict the "
        "weather condition with a confidence score."
    )

    predictor = get_predictor()

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            date = st.date_input("Date", value=datetime.date.today())
            precipitation = st.number_input(
                "Precipitation (mm)", min_value=0.0, max_value=100.0,
                value=0.0, step=0.1,
            )
            wind = st.number_input(
                "Wind Speed (m/s)", min_value=0.0, max_value=20.0,
                value=3.0, step=0.1,
            )

        with col2:
            temp_max = st.number_input(
                "Max Temperature (°C)", min_value=-20.0, max_value=50.0,
                value=20.0, step=0.5,
            )
            temp_min = st.number_input(
                "Min Temperature (°C)", min_value=-25.0, max_value=45.0,
                value=10.0, step=0.5,
            )

        submitted = st.form_submit_button("Predict Weather", width="stretch")

    if submitted:
        if temp_min > temp_max:
            st.error("Min temperature cannot be greater than max temperature.")
        else:
            result = predictor.predict(
                date=str(date),
                precipitation=precipitation,
                temp_max=temp_max,
                temp_min=temp_min,
                wind=wind,
            )

            st.session_state["last_prediction"] = result
            st.session_state["last_inputs"] = {
                "date": str(date),
                "precipitation": precipitation,
                "temp_max": temp_max,
                "temp_min": temp_min,
                "wind": wind,
            }

    if "last_prediction" in st.session_state:
        result = st.session_state["last_prediction"]
        icon = weather_icon(result["label"])

        col1, col2 = st.columns([1, 1.4])

        with col1:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-emoji">{icon}</div>
                    <div class="result-label">{result['label']}</div>
                    <p>Confidence: <b>{result['confidence']:.1%}</b></p>
                    <p style="color:#64748b;">Model: {result['model_used']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            proba_df = pd.DataFrame(
                sorted(result["probabilities"].items(), key=lambda x: x[1]),
                columns=["Weather", "Probability"],
            )
            fig = px.bar(
                proba_df,
                x="Probability",
                y="Weather",
                orientation="h",
                title="Prediction Confidence by Class",
                color="Probability",
                color_continuous_scale="Blues",
            )
            fig.update_layout(xaxis_tickformat=".0%", coloraxis_showscale=False)
            st.plotly_chart(fig, width="stretch")

        report_lines = [
            "AtmosIQ Prediction Report",
            "=" * 30,
            f"Date              : {st.session_state['last_inputs']['date']}",
            f"Precipitation     : {st.session_state['last_inputs']['precipitation']} mm",
            f"Max Temperature   : {st.session_state['last_inputs']['temp_max']} °C",
            f"Min Temperature   : {st.session_state['last_inputs']['temp_min']} °C",
            f"Wind Speed        : {st.session_state['last_inputs']['wind']} m/s",
            "-" * 30,
            f"Prediction        : {result['label']}",
            f"Confidence        : {result['confidence']:.2%}",
            f"Model Used        : {result['model_used']}",
            "-" * 30,
            "Class Probabilities:",
        ] + [
            f"  {k:<10}: {v:.2%}"
            for k, v in sorted(
                result["probabilities"].items(), key=lambda x: x[1], reverse=True
            )
        ]

        st.download_button(
            "⬇️ Download Prediction Report",
            data="\n".join(report_lines),
            file_name="atmosiq_prediction_report.txt",
            mime="text/plain",
            width="stretch",
        )


# ======================================================
# PAGE: Explainability
# ======================================================

elif page == "🧠 Explainability":

    st.header("🧠 Model Explainability (SHAP)")
    st.caption(
        "Understand which features drive the model's predictions — "
        "globally across the dataset, and locally for your last prediction."
    )

    explainer = get_explainer()

    tab1, tab2 = st.tabs(["Global Feature Importance", "Explain My Last Prediction"])

    with tab1:
        if st.button("Generate / Refresh Global SHAP Summary", width="stretch"):
            with st.spinner("Computing SHAP values — this can take a moment..."):
                explainer.plot_global_summary()

        if SHAP_BAR_PATH.exists():
            st.image(
                str(SHAP_BAR_PATH),
                caption=f"Global feature importance — {explainer.model_name}",
                width="stretch",
            )
        else:
            st.info("Click the button above to generate the SHAP summary plot.")

    with tab2:
        if "last_inputs" not in st.session_state:
            st.info(
                "Make a prediction on the '🔮 Make a Prediction' page first, "
                "then come back here to see why the model decided what it did."
            )
        else:
            inputs = st.session_state["last_inputs"]
            result = st.session_state["last_prediction"]

            st.write(
                f"Explaining the prediction **{result['label']}** "
                f"({result['confidence']:.1%} confidence) for the inputs "
                f"you last submitted."
            )

            if st.button("Explain This Prediction", width="stretch"):
                with st.spinner("Computing local SHAP explanation..."):
                    row = build_feature_row(
                        date=inputs["date"],
                        precipitation=inputs["precipitation"],
                        temp_max=inputs["temp_max"],
                        temp_min=inputs["temp_min"],
                        wind=inputs["wind"],
                    )
                    impact = explainer.explain_instance(row)

                impact_df = pd.DataFrame(
                    impact.items(), columns=["Feature", "SHAP Value"]
                )
                impact_df["Direction"] = impact_df["SHAP Value"].apply(
                    lambda v: "Pushes toward" if v > 0 else "Pushes away from"
                )

                fig = px.bar(
                    impact_df.sort_values("SHAP Value"),
                    x="SHAP Value",
                    y="Feature",
                    orientation="h",
                    color="SHAP Value",
                    color_continuous_scale="RdBu",
                    title=f"Why the model predicted '{result['label']}'",
                )
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig, width="stretch")

                st.caption(
                    "Positive bars push the prediction toward the predicted "
                    "class; negative bars push away from it."
                )


# ======================================================
# PAGE: Prediction History
# ======================================================

elif page == "🗂️ Prediction History":

    st.header("🗂️ Prediction History")

    db = get_db()
    history = db.get_prediction_history()
    db.close()

    if history.empty:
        st.info("No predictions logged yet — make one on the Prediction page.")
    else:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.dataframe(history, width="stretch", hide_index=True)

        with col2:
            counts = history["prediction"].value_counts().reset_index()
            counts.columns = ["prediction", "count"]
            fig = px.pie(
                counts,
                names="prediction",
                values="count",
                title="Prediction Breakdown",
                hole=0.45,
            )
            st.plotly_chart(fig, width="stretch")

        avg_confidence = history["confidence"].mean()
        st.metric("Average Confidence Across All Predictions", f"{avg_confidence:.1%}")

        st.download_button(
            "⬇️ Download Full History (CSV)",
            data=history.to_csv(index=False),
            file_name="atmosiq_prediction_history.csv",
            mime="text/csv",
            width="stretch",
        )