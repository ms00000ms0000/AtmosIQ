<div align="center">

# 🌦️ AtmosIQ
### AI-Powered Multi-Class Weather Prediction System using Machine Learning, Deep Learning & Explainable AI

<p align="center">
An enterprise-grade weather prediction platform that compares multiple Machine Learning algorithms, selects the best performing model using Cross Validation, explains predictions using SHAP Explainable AI, and provides real-time weather forecasting through an interactive Streamlit web application.
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![SHAP](https://img.shields.io/badge/Explainable-AI-blueviolet?style=for-the-badge)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)

</p>

<p align="center">

<a href="https://atmosiq.streamlit.app/">
<img src="https://img.shields.io/badge/🚀%20Live%20Demo-Visit%20WebApp-success?style=for-the-badge">
</a>

<a href="https://github.com/ms00000ms0000/AtmosIQ">
<img src="https://img.shields.io/badge/📂%20Source%20Code-GitHub-black?style=for-the-badge">
</a>

</p>

</div>

---

# 📌 Table of Contents

- Project Overview
- Live Demo
- Architecture
- Features
- Project Highlights
- Machine Learning Pipeline
- Deep Learning Pipeline
- Explainable AI (SHAP)
- Database Integration
- Technology Stack
- Project Workflow
- Screenshots
- Folder Structure
- Installation
- Usage
- Model Training
- Prediction Pipeline
- Model Comparison
- API Endpoints
- Performance Metrics
- Future Improvements
- Author
- License

---

# 🚀 Live Demo

### 🌐 Streamlit Application

> https://atmosiq.streamlit.app/

Experience real-time weather prediction directly from your browser.

The application allows users to:

- Predict weather conditions instantly
- Compare multiple trained ML models
- View prediction confidence
- Explore feature importance using SHAP Explainability
- Analyze model performance
- Visualize prediction probabilities
- Interact with a production-ready AI interface

---

# 📖 Project Overview

AtmosIQ is a production-inspired **AI-powered Weather Prediction System** designed to classify weather conditions from meteorological attributes using both **Machine Learning** and **Deep Learning** approaches.

Instead of relying on a single algorithm, AtmosIQ evaluates multiple state-of-the-art supervised learning models through a robust **5-Fold Stratified Cross Validation** pipeline, automatically selecting the best-performing model for deployment.

The project follows an end-to-end Machine Learning lifecycle including:

- Data preprocessing
- Feature engineering
- Model training
- Cross-validation
- Hyperparameter optimization
- Explainable AI
- Model persistence
- SQL database integration
- Interactive Streamlit deployment

Unlike traditional academic ML projects, AtmosIQ emphasizes software engineering best practices by separating training, prediction, visualization, database operations, and deployment into modular components.

---

# 🎯 Problem Statement

Weather prediction plays a critical role across multiple industries including:

- Agriculture
- Aviation
- Disaster Management
- Smart Cities
- Logistics
- Tourism
- Transportation

Traditional forecasting methods often require complex meteorological infrastructure.

AtmosIQ demonstrates how Machine Learning can leverage historical weather observations to accurately classify future weather conditions while providing transparency through Explainable AI techniques.

---

# ✨ Key Features

## 🤖 Multiple Machine Learning Models

Instead of depending on one algorithm, AtmosIQ trains and compares multiple models:

- Gradient Boosting Classifier
- Random Forest Classifier
- Hist Gradient Boosting
- Extra Trees Classifier
- Neural Network (TensorFlow/Keras)

The best-performing model is selected automatically using validation metrics.

---

## 📊 Automated Model Comparison

The system automatically compares all trained models using:

- Accuracy
- Precision
- Recall
- F1 Score
- Cross Validation Score
- Confusion Matrix
- Classification Report

Results are stored for future analysis.

---

## 🧠 Explainable AI

AtmosIQ integrates **SHAP (SHapley Additive Explanations)** to make predictions interpretable.

Instead of behaving like a black box, the application explains:

- Feature importance
- Local prediction impact
- Global model behavior
- Decision transparency

This improves trust and interpretability for end users.

---

## 🔄 5-Fold Stratified Cross Validation

Every Machine Learning model is evaluated using:

- Stratified K-Fold Cross Validation
- Balanced class distribution
- Better generalization
- Reduced overfitting
- Reliable performance estimation

---

## 💾 Model Persistence

After training, all important artifacts are saved automatically for production inference.

Saved artifacts include:

- `best_model.pkl`
- `weather_nn.keras`
- `best_checkpoint.keras`
- `best_model_name.txt`
- `season_encoder.pkl`
- `target_encoder.pkl`
- `scaler.pkl`
- `model_comparison.csv`

This enables fast loading during inference without retraining.

---

## 🗄 SQL Database Integration

AtmosIQ uses SQL-based storage to manage application data.

The database layer can be used for:

- Prediction history
- User inputs
- Model metadata
- Analytics
- Logging
- Historical weather predictions

This makes the application more scalable compared to stateless prediction systems.

---

## 🌐 Interactive Streamlit Dashboard

A production-ready Streamlit interface enables users to:

- Enter weather parameters
- Receive predictions instantly
- Compare models
- View confidence scores
- Understand predictions through SHAP
- Explore interactive visualizations

---

## 📈 Enterprise Project Structure

The project follows a modular architecture inspired by production ML systems.

Major modules include:

- Data Pipeline
- Training Engine
- Prediction Engine
- Explainability
- Database Layer
- Visualization Layer
- Deployment Layer

This structure improves maintainability, scalability, and readability.

---

# 🏆 Project Highlights

✅ End-to-End Machine Learning Project

✅ Production-Inspired Architecture

✅ Multi-Model Training

✅ Deep Learning Integration

✅ Explainable AI using SHAP

✅ SQL Database Support

✅ 5-Fold Cross Validation

✅ Automatic Best Model Selection

✅ Streamlit Deployment

✅ Professional Folder Structure

✅ Saved Production Artifacts

✅ Modular Python Codebase

✅ Recruiter-Friendly Documentation

---

# 🧠 Machine Learning Models Used

| Model | Purpose |
|--------|----------|
| Gradient Boosting | Ensemble Learning |
| Random Forest | Bagging Ensemble |
| Hist Gradient Boosting | Histogram-based Gradient Boosting |
| Extra Trees | Extremely Randomized Trees |
| Neural Network (TensorFlow) | Deep Learning Classification |

Each model is independently trained, evaluated, and compared before selecting the final production model.

---

# 📊 Prediction Classes

AtmosIQ is capable of predicting multiple weather conditions, including categories such as:

- ☀️ Sunny
- 🌧 Rain
- 🌫 Fog
- ❄ Snow
- 🌦 Drizzle

The final predicted class depends on the trained model and processed weather features.

---

# 🏗️ System Architecture

<p align="center">

> Replace the path below with your uploaded architecture image.

```text
images/Atmos_Architecture.png
```

<img src="images/Atmos_Architecture.png" width="100%">

</p>

The complete architecture illustrates the end-to-end flow of the system, beginning with raw weather data ingestion, followed by preprocessing, feature engineering, model training, explainability, database operations, prediction services, and deployment through Streamlit.

---

# 📸 Project Screenshots

> Replace the image paths below with your uploaded screenshots.

## 🏠 Home Page

```text
screenshots/home.png
```

## 📊 Dashboard

```text
screenshots/dashboard.png
```

## 🤖 Prediction

```text
screenshots/prediction.png
```

## 📈 Model Comparison

```text
screenshots/model_comparison.png
```

## 📉 SHAP Explainability

```text
screenshots/shap.png
```

---

---

# 🛠️ Technology Stack

AtmosIQ is built using a modern Machine Learning stack inspired by production-grade AI applications.

| Category | Technologies |
|----------|--------------|
| Programming Language | Python 3.10+ |
| Machine Learning | Scikit-Learn |
| Deep Learning | TensorFlow / Keras |
| Explainable AI | SHAP |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Plotly |
| Web Framework | Streamlit |
| Database | SQLite (SQL) |
| Model Serialization | Joblib |
| Cross Validation | Stratified K-Fold |
| Version Control | Git & GitHub |
| Deployment | Streamlit Cloud |

---

# 🧩 Technology Overview

## Programming Language

- Python 3.10+

AtmosIQ is entirely developed using Python because of its powerful ecosystem for Machine Learning, Data Science, Deep Learning, and rapid deployment.

---

## Machine Learning

Scikit-Learn powers the complete ML pipeline including:

- Feature preprocessing
- Model training
- Model evaluation
- Cross Validation
- Metrics
- Pipelines
- Model persistence

---

## Deep Learning

TensorFlow/Keras is used for developing the Neural Network model.

Capabilities include:

- Dense Neural Networks
- Checkpoint Saving
- Early Stopping
- Model Serialization
- Production Inference

---

## Explainable AI

SHAP (SHapley Additive Explanations) provides transparency into model predictions.

Features:

- Feature Importance
- Local Explanations
- Global Explanations
- Model Interpretability

---

## Database

SQLite stores structured application data including:

- Prediction History
- User Inputs
- Logs
- Analytics
- Model Metadata

---

## Deployment

The application is deployed using Streamlit Cloud for public access.

Deployment Features:

- Browser-based Interface
- Real-time Prediction
- No Installation Required
- Cloud Hosted

---

# ⚙️ Project Workflow

The following workflow represents the complete lifecycle of AtmosIQ.

```text
                     Weather Dataset
                            │
                            ▼
                  Data Preprocessing
                            │
                            ▼
                  Feature Engineering
                            │
                            ▼
                Feature Encoding & Scaling
                            │
                            ▼
              Train / Validation / Test Split
                            │
                            ▼
          ┌────────────────────────────────────┐
          │                                    │
          ▼                                    ▼
Machine Learning Models                 Neural Network
          │                                    │
          └──────────────┬─────────────────────┘
                         ▼
             5-Fold Cross Validation
                         │
                         ▼
                Performance Evaluation
                         │
                         ▼
                Best Model Selection
                         │
                         ▼
                SHAP Explainability
                         │
                         ▼
                Save Production Artifacts
                         │
                         ▼
                  SQLite Database
                         │
                         ▼
                 Streamlit Application
                         │
                         ▼
                 Real-Time Prediction
```

---

# 📂 Project Directory Structure

```text
AtmosIQ
│
├── app/
│   ├── pages/
│   ├── components/
│   ├── utils/
│   └── app.py
│
├── artifacts/
│   ├── best_model.pkl
│   ├── weather_nn.keras
│   ├── best_checkpoint.keras
│   ├── best_model_name.txt
│   ├── scaler.pkl
│   ├── season_encoder.pkl
│   ├── target_encoder.pkl
│   └── model_comparison.csv
│
├── database/
│   └── SQLite Database Files
│
├── models/
│   ├── gradient_boosting.py
│   ├── random_forest.py
│   ├── hist_gradient_boosting.py
│   ├── extra_trees.py
│   └── neural_network.py
│
├── notebooks/
│
├── reports/
│
├── screenshots/
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── database.py
│   ├── explainability.py
│   ├── utils.py
│   └── config.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

> **Note:** The exact structure may vary depending on future updates to the repository.

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/ms00000ms0000/AtmosIQ.git
```

Move into the project directory

```bash
cd AtmosIQ
```

Create a virtual environment

### Windows

```bash
python -m venv venv
```

Activate environment

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

# 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Streamlit Application

```bash
streamlit run app.py
```

or

```bash
python -m streamlit run app.py
```

---

# 🧠 Train the Models

To retrain every model from scratch:

```bash
python src/train.py
```

Training automatically performs:

- Data Cleaning
- Feature Engineering
- Feature Encoding
- Scaling
- Model Training
- Neural Network Training
- Cross Validation
- Model Comparison
- SHAP Generation
- Artifact Saving

---

# 🔍 Make Predictions

Run prediction module

```bash
python src/predict.py
```

The prediction engine automatically loads:

- Best Model
- Encoders
- Scaler
- Neural Network (if selected)
- Configuration Files

without requiring retraining.

---

# 🧪 Machine Learning Pipeline

The training pipeline consists of multiple modular stages.

### 1. Data Collection

- Historical Weather Dataset
- Structured CSV Input

↓

### 2. Data Cleaning

- Missing Values
- Invalid Entries
- Duplicate Removal

↓

### 3. Feature Engineering

- Weather Features
- Season Encoding
- Numerical Transformations

↓

### 4. Encoding

Categorical features are transformed using trained encoders.

Artifacts saved:

- season_encoder.pkl
- target_encoder.pkl

↓

### 5. Feature Scaling

Numerical variables are normalized using StandardScaler.

Artifact:

```
scaler.pkl
```

↓

### 6. Train-Test Split

Dataset is divided using Stratified Sampling.

↓

### 7. Train Multiple Models

- Gradient Boosting
- Random Forest
- Hist Gradient Boosting
- Extra Trees
- Neural Network

↓

### 8. Cross Validation

Each model undergoes

- Stratified K-Fold
- 5 folds
- Average performance calculation

↓

### 9. Performance Evaluation

Metrics include

- Accuracy
- Precision
- Recall
- F1 Score

↓

### 10. Best Model Selection

Highest-performing model is automatically selected.

↓

### 11. Save Artifacts

Artifacts generated:

```text
best_model.pkl
weather_nn.keras
best_checkpoint.keras
best_model_name.txt
season_encoder.pkl
target_encoder.pkl
scaler.pkl
model_comparison.csv
```

---

# ⚡ Prediction Pipeline

The deployed prediction pipeline follows these steps:

```text
User Input
      │
      ▼
Input Validation
      │
      ▼
Feature Encoding
      │
      ▼
Scaling
      │
      ▼
Load Best Model
      │
      ▼
Prediction
      │
      ▼
Probability Calculation
      │
      ▼
SHAP Explanation
      │
      ▼
Prediction Display
      │
      ▼
Store in SQL Database
```

---

# 🗄️ Database Workflow

SQLite is integrated into AtmosIQ to organize application-level information.

The database stores:

- Prediction History
- Weather Inputs
- Model Used
- Prediction Confidence
- Timestamp
- Application Logs
- Analytical Information

Workflow:

```text
User Input

↓

Prediction

↓

Insert Record

↓

Store in SQLite

↓

Dashboard Analytics

↓

Future Analysis
```

---

# 🧠 Explainable AI Workflow

AtmosIQ integrates SHAP to provide transparency for every prediction.

Workflow:

```text
Model Prediction

↓

Load SHAP Explainer

↓

Calculate SHAP Values

↓

Feature Contribution

↓

Visual Explanation

↓

User Interpretation
```

Benefits include:

- Transparent Predictions

- Improved Trust

- Better Model Understanding

- Feature Impact Analysis

- Explainable Decision Making

---

# 📦 Production Artifacts

During training, AtmosIQ automatically generates reusable production artifacts.

| Artifact | Purpose |
|-----------|----------|
| best_model.pkl | Best Machine Learning Model |
| weather_nn.keras | Neural Network Model |
| best_checkpoint.keras | Best Neural Network Checkpoint |
| scaler.pkl | Feature Scaling |
| season_encoder.pkl | Season Encoding |
| target_encoder.pkl | Target Label Encoding |
| best_model_name.txt | Stores Selected Model Name |
| model_comparison.csv | Performance Comparison |

These artifacts eliminate the need to retrain the models before deployment.

---
---

# 📊 Model Comparison Strategy

Unlike conventional Machine Learning projects that rely on a single algorithm, AtmosIQ follows a competitive model evaluation approach where multiple algorithms are trained, validated, and compared before selecting the best-performing model.

## Models Evaluated

| Model | Type | Purpose |
|--------|------|----------|
| 🌳 Gradient Boosting | Ensemble Learning | Sequential boosting of weak learners |
| 🌲 Random Forest | Bagging Ensemble | Robust classification with reduced variance |
| 🌿 Hist Gradient Boosting | Histogram-based Boosting | Faster gradient boosting for large datasets |
| 🌴 Extra Trees | Randomized Ensemble | Improved generalization through random splits |
| 🧠 Neural Network | Deep Learning | Non-linear pattern learning using TensorFlow |

---

## Model Evaluation Metrics

Each model is evaluated using the following performance metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- 5-Fold Cross Validation Score
- Classification Report
- Confusion Matrix

The model with the best validation performance is automatically selected as the production model.

---

## Cross Validation

AtmosIQ performs **5-Fold Stratified Cross Validation** to ensure robust model evaluation.

### Advantages

- Better Generalization
- Reduced Overfitting
- Stable Performance Estimation
- Balanced Class Distribution
- Fair Model Comparison

This strategy provides a more reliable estimate of real-world model performance compared to a single train-test split.

---

# 📈 Performance Evaluation

The evaluation pipeline includes multiple stages to comprehensively assess model performance.

### Classification Metrics

- Accuracy
- Precision
- Recall
- F1 Score

### Validation Metrics

- Cross Validation Mean Score
- Cross Validation Standard Deviation

### Visual Analysis

- Confusion Matrix
- Feature Importance
- SHAP Summary Plot
- Model Comparison Table

The generated reports help identify strengths and weaknesses of each algorithm before deployment.

---

# 🧠 Explainable AI using SHAP

Modern AI systems should not only make accurate predictions but also explain **why** those predictions were made.

AtmosIQ integrates **SHAP (SHapley Additive Explanations)** to provide transparent and interpretable predictions.

## Why SHAP?

Traditional Machine Learning models often act as black boxes, making it difficult to understand the reasoning behind predictions.

SHAP addresses this by quantifying the contribution of each feature toward the final prediction.

### SHAP Features

- Global Feature Importance
- Local Prediction Explanation
- Feature Contribution Analysis
- Model Transparency
- Decision Interpretability

---

## SHAP Workflow

```text
Weather Features
        │
        ▼
Trained Model
        │
        ▼
SHAP Explainer
        │
        ▼
Calculate SHAP Values
        │
        ▼
Feature Contribution Scores
        │
        ▼
Visual Explanation
```

---

## Benefits of Explainable AI

- Improves trust in predictions
- Helps understand model behavior
- Identifies influential weather parameters
- Supports debugging and model refinement
- Makes AI decisions interpretable for end users

---

# 🖥️ Streamlit Web Application

AtmosIQ is deployed as an interactive web application using **Streamlit**, allowing users to make weather predictions in real time without requiring any local setup.

### Key Features

- User-friendly interface
- Responsive design
- Instant weather prediction
- Model confidence display
- SHAP-based explainability
- Model comparison
- Visual analytics
- Fast prediction response

---

# 🌐 Live Application

**🔗 Live Demo**

https://atmosiq.streamlit.app/

---

# 💡 Application Workflow

```text
Launch Streamlit App
        │
        ▼
Enter Weather Parameters
        │
        ▼
Input Validation
        │
        ▼
Preprocessing
        │
        ▼
Load Best Model
        │
        ▼
Generate Prediction
        │
        ▼
Prediction Probability
        │
        ▼
SHAP Explanation
        │
        ▼
Display Final Result
```

---

# 📡 Prediction Process

When a user submits weather parameters:

1. Input values are validated.
2. Features are encoded.
3. Numerical values are scaled.
4. The best trained model is loaded.
5. Weather condition is predicted.
6. Confidence score is calculated.
7. SHAP explanation is generated.
8. Prediction is displayed.
9. Prediction history is stored in the SQL database.

---

# ⚡ Why AtmosIQ?

AtmosIQ goes beyond a standard Machine Learning project by combining:

- Classical Machine Learning
- Deep Learning
- Explainable AI
- SQL Database Integration
- Production Artifacts
- Cross Validation
- Interactive Deployment
- Modular Codebase

This makes it suitable as a portfolio project demonstrating both Data Science and Machine Learning Engineering skills.

---

# 🚀 Future Enhancements

Potential future improvements include:

- 🌦️ Live Weather API Integration
- 📍 Location-Based Weather Prediction
- ☁️ Cloud Deployment on AWS / Azure / GCP
- 🐳 Docker Containerization
- ☸️ Kubernetes Deployment
- 🔄 CI/CD Pipeline using GitHub Actions
- 📊 Advanced Analytics Dashboard
- 📱 Mobile Application
- 🤖 AutoML Integration
- 🔔 Weather Alerts & Notifications
- 🌍 Multi-language Support
- 📈 Continuous Model Retraining

---

# 🤝 Contributing

Contributions are welcome!

If you'd like to contribute:

1. Fork the repository
2. Create a new feature branch

```bash
git checkout -b feature-name
```

3. Commit your changes

```bash
git commit -m "Add new feature"
```

4. Push your branch

```bash
git push origin feature-name
```

5. Open a Pull Request

Every contribution, whether it's fixing bugs, improving documentation, or adding new features, is greatly appreciated.

---

# ⭐ Support the Project

If you found this project useful:

- ⭐ Star the repository
- 🍴 Fork the project
- 🐛 Report issues
- 💡 Suggest new features
- 📢 Share it with others

Your support helps improve the project and motivates future development.

---

# 👨‍💻 About the Developer

<div align="center">

## **Mayank Srivastava**

**AI Engineer | Machine Learning Engineer | Android Developer | Full Stack Developer**

Passionate about building production-ready AI systems, scalable Machine Learning solutions, and modern full-stack applications. My work focuses on solving real-world problems through intelligent software, combining clean engineering practices with practical deployment.

</div>

---

# 🌐 Connect with Me

<p align="center">

<a href="https://github.com/ms00000ms0000">
<img src="https://img.shields.io/badge/GitHub-ms00000ms0000-black?style=for-the-badge&logo=github">
</a>

<a href="https://www.linkedin.com/in/ms8960/">
<img src="https://img.shields.io/badge/LinkedIn-Mayank%20Srivastava-blue?style=for-the-badge&logo=linkedin">
</a>

<a href="https://meetms.netlify.app/">
<img src="https://img.shields.io/badge/Portfolio-Visit-success?style=for-the-badge&logo=googlechrome">
</a>

<a href="mailto:msrivastava194@gmail.com">
<img src="https://img.shields.io/badge/Email-Contact-red?style=for-the-badge&logo=gmail">
</a>

<a href="https://www.instagram.com/ms0000ms0000">
<img src="https://img.shields.io/badge/Instagram-Follow-E4405F?style=for-the-badge&logo=instagram&logoColor=white">
</a>

</p>

---

# 📬 Contact

**Author:** Mayank Srivastava

📧 Email: **msrivastava194@gmail.com**

🌐 Portfolio: **https://meetms.netlify.app/**

💼 LinkedIn: **https://www.linkedin.com/in/ms8960/**

💻 GitHub: **https://github.com/ms00000ms0000**

📸 Instagram: **https://www.instagram.com/ms0000ms0000**

---

# 📄 License

This project is licensed under the **MIT License**.

You are free to use, modify, and distribute this project in accordance with the license terms.

See the `LICENSE` file for more details.

---

# 🙏 Acknowledgements

Special thanks to the open-source community and the creators of the amazing tools and libraries that made this project possible:

- Python
- Scikit-Learn
- TensorFlow
- Streamlit
- SHAP
- Pandas
- NumPy
- Matplotlib
- Plotly
- SQLite
- Joblib
- Git & GitHub

Their contributions continue to empower developers and researchers around the world.

---

# ⭐ If you like this project...

<div align="center">

### Give it a ⭐ on GitHub!

Your support encourages further development and helps others discover the project.

---

### 🚀 Built with ❤️ by **Mayank Srivastava**

**AtmosIQ — AI-Powered Weather Prediction System**

*From raw weather data to explainable, production-ready predictions.*

</div>
