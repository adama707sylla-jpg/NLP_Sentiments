from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import re

from config import MODEL_PATH, PROJECT_NAME

# Charger le modèle
app   = FastAPI(title=f"API {PROJECT_NAME}")
model = joblib.load(MODEL_PATH)

# ── Fonction nettoyage
def nettoyer_texte(texte):
    texte = str(texte).lower()
    texte = re.sub(r'http\S+', '', texte)
    texte = re.sub(r'@\w+',    '', texte)
    texte = re.sub(r'#\w+',    '', texte)
    texte = re.sub(r'[^a-z\s]', ' ', texte)
    texte = re.sub(r'\s+',     ' ', texte).strip()
    return texte

# Modèles Pydantic
class TextInput(BaseModel):
    text: str

class BatchInput(BaseModel):
    texts: List[str]

# Endpoints

@app.get("/")
def accueil():
    return {
        "status"  : "ok",
        "projet"  : PROJECT_NAME,
        "modele"  : "LogisticRegression + TF-IDF",
        "endpoints": ["/predict", "/predict_batch", "/docs"]
    }

@app.post("/predict")
def predire(data: TextInput):
    # 1. Nettoyer le texte
    texte_clean = nettoyer_texte(data.text)

    # 2. Prédire
    prediction = model.predict([texte_clean])[0]
    proba      = model.predict_proba([texte_clean])[0].max()

    return {
        "sentiment" : prediction,
        "confiance" : round(float(proba), 4),
        "texte_recu": data.text[:100]   # aperçu du texte reçu
    }

@app.post("/predict_batch")
def predire_batch(data: BatchInput):
    # Nettoyer tous les textes
    textes_clean = [nettoyer_texte(t) for t in data.texts]

    # Prédire en une seule fois
    predictions = model.predict(textes_clean)
    probas      = model.predict_proba(textes_clean).max(axis=1)

    resultats = []
    for texte, pred, proba in zip(data.texts, predictions, probas):
        resultats.append({
            "texte"    : texte[:80],
            "sentiment": pred,
            "confiance": round(float(proba), 4)
        })

    return {
        "total"      : len(resultats),
        "predictions": resultats
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}