# Churn Prediction System

A complete customer-churn prediction project — from data preprocessing and model training, to deployment via a Streamlit web app. This project demonstrates building a full-pipeline ML solution with feature engineering, model comparison, and a user-friendly interface for real-time predictions.

## Purpose

The goal of this project is to predict whether a customer is likely to churn (cancel service) using historical telecom data. By training and comparing several models, and then deploying the best one, this system can help businesses identify at-risk customers and intervene proactively — improving retention and reducing revenue loss.

## Repository Structure
```
Churn-Prediction-System/
│
├── app.py # Streamlit application to input customer data & predict churn
├── requirements.txt # Python dependencies
│
├── models/ # Saved artifacts for deployment
│ ├── xgb_churn_model.pkl # Trained XGBoost model
│ ├── scaler.pkl # StandardScaler used to scale numeric features
│ └── feature_names.pkl # List of all feature column names after preprocessing
│
├── notebooks/ # (Optional) Notebooks used for EDA and model development
│ └── churn_modeling.ipynb # Jupyter notebook with data cleaning, EDA, model training & evaluation
```
## Results

| Model | Accuracy | Recall (Churn = 1) | Comments |
|-------|----------|--------------------|----------|
| Logistic Regression | ~72% | **81%** | Good at catching churners (high recall) |
| Random Forest | ~78% | 50% | Strong overall accuracy but misses many churn cases |
| XGBoost (final) | ~75% | 78% | Balanced performance; selected as production model |

The final XGBoost model was chosen because it offered a strong trade-off between precision, recall, and overall stability after cross-validation.

## How to Run the Project Locally

# 1. Clone the repo
```bash
git clone https://github.com/YourUsername/Churn-Prediction-System.git  
cd Churn-Prediction-System
```
# 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate   # (Windows: venv\\Scripts\\activate)
```
# 3. Install dependencies
```bash
pip install -r requirements.txt
```
# 4. Run the Streamlit app
```bash
streamlit run app.py
```
Then open your browser at http://localhost:8501 to access the web interface. You can input customer details (tenure, monthly charges, contract type, etc.) and get a churn prediction with probability.

# How It Works

Numeric features (tenure, MonthlyCharges, TotalCharges) are standardized using the saved scaler.pkl.

Categorical features (Contract, PaymentMethod, etc.) are one-hot encoded, and the resulting feature vector is aligned via feature_names.pkl to match the model’s expected input.

The trained XGBoost model in xgb_churn_model.pkl makes the prediction — either churn (1) or no churn (0) — and returns a probability score.

# What You Can Do

Use the Streamlit app to experiment with different customer profiles and study which factors influence churn.

Retrain the model on new data (update dataset, preprocessing, or model hyperparameters) to improve performance.

Integrate the model into a production environment (e.g. as a REST API) instead of using Streamlit — making it accessible from other services.

# Demo

A video walkthrough of this deployed project will be added here soon — showcasing:

Using the Streamlit app

Example predictions

Explanation of predictions (input → output)

Model performance summary and comparison