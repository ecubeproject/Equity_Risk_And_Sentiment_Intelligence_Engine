# api/app/inference.py
import numpy as np

# -----------------------------------------------------------------------------  
# Helper – keep feature extraction in one place
# -----------------------------------------------------------------------------
def _vectorize(text: str, vectorizer):
    """Transform raw headline into the TF-IDF feature space used at training."""
    return vectorizer.transform([text])

# -----------------------------------------------------------------------------  
# Sentiment
# -----------------------------------------------------------------------------
def predict_sentiment(text: str, model, vectorizer) -> int:
    """
    Hard-label sentiment prediction.
    Returns an integer class index (0, 1, 2, …) that you later
    convert to a string label with the LabelEncoder.
    """
    X = _vectorize(text, vectorizer)
    pred = model.predict(X)
    return int(pred[0])

# -----------------------------------------------------------------------------  
# Risk
# -----------------------------------------------------------------------------
def predict_risk(text: str, model, vectorizer) -> float:
    """
    Return **probability** that the headline is high-risk (class 1).

    The caller (main.py) decides whether that probability crosses the tuned
    threshold stored in `risk_cutoff.npy`, so we **do no thresholding here**.
    """
    X = _vectorize(text, vectorizer)

    # XGBoost’s `predict_proba` -> shape (1, 2); column 1 = P(class 1 | X)
    prob_pos = float(model.predict_proba(X)[0, 1])
    return prob_pos
