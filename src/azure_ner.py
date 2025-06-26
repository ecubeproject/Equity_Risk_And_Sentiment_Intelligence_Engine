# ── src/azure_ner.py ──────────────────────────────────────────────────────────
import os, requests

API_VERSION = os.getenv("AZURE_LANG_API_VERSION", "2022-05-01")

def ner(texts: list[str]) -> list[dict]:
    """Low-level call to Azure AI Language – Named-Entity Recognition."""
    endpoint = os.getenv("AZURE_LANG_ENDPOINT")
    key      = os.getenv("AZURE_LANG_KEY")
    if not (endpoint and key):
        raise RuntimeError("Missing AZURE_LANG_ENDPOINT or AZURE_LANG_KEY")

    url = f"{endpoint}/language/:analyze-text?api-version={API_VERSION}"
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "application/json",
    }

    docs = [{"id": str(i + 1), "language": "en", "text": t}
            for i, t in enumerate(texts)]

    body = {
        "kind": "EntityRecognition",
        "analysisInput": {"documents": docs},
        "parameters": {"modelVersion": "latest"},
    }

    resp = requests.post(url, headers=headers, json=body, timeout=10)
    resp.raise_for_status()
    return resp.json()["results"]["documents"]           # ← Azure format


# ---------- safe façade -------------------------------------------------------
def azure_ner(texts: list[str]) -> list[dict]:
    """
    Wrapper used by FastAPI.  Always returns a list; on failure embeds the error
    instead of crashing the whole request handler.
    """
    try:
        return ner(texts)
    except Exception as exc:
        return [{"error": str(exc)}]
