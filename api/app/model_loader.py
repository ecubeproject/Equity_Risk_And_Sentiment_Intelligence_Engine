# api/app/model_loader.py
import os, joblib, numpy as np

def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(base_dir))

    sentiment_model_path = os.path.join(project_root, "models", "xgboost_sentiment_model.joblib")
    risk_model_path      = os.path.join(project_root, "models", "risk_model_bal.joblib")
    vectorizer_path      = os.path.join(project_root, "models", "vectorizer.pkl")
    label_encoder_path   = os.path.join(project_root, "models", "label_encoder.pkl")
    cutoff_path          = os.path.join(project_root, "models", "risk_cutoff.npy")

    sentiment_model = joblib.load(sentiment_model_path)
    risk_model      = joblib.load(risk_model_path)
    vectorizer      = joblib.load(vectorizer_path)
    label_encoder   = joblib.load(label_encoder_path)
    risk_cutoff     = float(np.load(cutoff_path))  # ensure scalar

    return sentiment_model, vectorizer, label_encoder, risk_model, risk_cutoff
