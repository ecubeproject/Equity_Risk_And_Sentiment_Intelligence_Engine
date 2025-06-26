# api/app/main.py
"""
Main FastAPI entry-point
───────────────────────────────────────────────────────────────────────────────
• Loads sentiment + risk models (local, XGBoost).
• Optionally calls Azure AI Language Named-Entity-Recognition (NER) if
  the two env-vars are present:
      AZURE_LANG_ENDPOINT   e.g. https://your-resource.cognitiveservices.azure.com
      AZURE_LANG_KEY        the primary / secondary key from the portal
"""

import os
from typing import Any, Dict

from dotenv import load_dotenv

from fastapi import FastAPI
from pydantic import BaseModel

from api.app.model_loader import load_model
from api.app.inference import predict_sentiment, predict_risk

# ── 1. Environment ----------------------------------------------------------------
load_dotenv()                       # Loads .env if it exists (harmless if not)

USE_AZURE_NER: bool = all(
    k in os.environ for k in ("AZURE_LANG_ENDPOINT", "AZURE_LANG_KEY")
)

if USE_AZURE_NER:
    from src.azure_ner import azure_ner
else:
    azure_ner = None      # dummy so the rest of the code still runs

print(f"[BOOT] USE_AZURE_NER = {USE_AZURE_NER}")

# ── 2. FastAPI setup ---------------------------------------------------------------
app = FastAPI(
    title="Sentiment & Risk Intelligence Engine",
    version="1.0",
    description="Predicts sentiment + risk, and (optionally) extracts entities via Azure AI Language.",
)

class HeadlineInput(BaseModel):
    title: str

# ── 3. Load ML artefacts -----------------------------------------------------------
sentiment_model, vectorizer, label_encoder, risk_model, risk_cutoff = load_model()

# ── 4. Helper – thin wrapper around azure_ner to keep main logic tidy -------------
def _call_azure_ner(headline: str) -> Dict[str, Any]:
    """Return either {'entities': [...]} or {'error': 'msg'} for graceful degradation."""
    try:
        if not USE_AZURE_NER:
            return {"note": "Azure NER not enabled"}
        resp = azure_ner([headline])          # -> list[dict]
        return resp[0] if isinstance(resp, list) else {"error": "Unexpected NER payload"}
    except Exception as exc:                  # network issues, 4xx, 5xx, …
        return {"error": str(exc)}

# ── 5. End-points ------------------------------------------------------------------
@app.post("/predict")
def predict(input: HeadlineInput):
    idx          = predict_sentiment(input.title, sentiment_model, vectorizer)
    sentiment    = label_encoder.inverse_transform([idx])[0]
    entities_out = _call_azure_ner(input.title)

    return {
        "headline"       : input.title,
        "predicted_class": sentiment,
        "entities"       : entities_out,
    }

@app.post("/predict_risk")
def predict_risk_flag(input: HeadlineInput):
    prob          = predict_risk(input.title, risk_model, vectorizer)
    flag          = prob >= risk_cutoff
    entities_out  = _call_azure_ner(input.title)

    return {
        "headline"        : input.title,
        "risk_probability": round(prob, 3),
        "risk_flag"       : bool(flag),
        "entities"        : entities_out,
    }

@app.get("/")
def root():
    return {
        "message": (
            "Welcome to the Sentiment Intelligence API — browse /docs for Swagger UI."
        )
    }
