import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

CONFIG = {
    "mode": "nlp",   # "nlp" | "classification" | "regression"

    # Pour NLP
    "nlp": {
        "texte_positif" : "This product is absolutely amazing, I love it!",
        "texte_negatif" : "Terrible product, complete waste of money!",
        "texte_neutre"  : "Good product",
        "batch_textes"  : ["Amazing!", "Terrible!", "It was okay"],
        "seuil_confiance": 0.85,
    }
}

MODE = CONFIG["mode"]

# TESTS


def test_root():
    """API vivante et repond"""
    r = client.get("/")
    assert r.status_code == 200

def test_health():
    """Modele bien charge en memoire"""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["model_loaded"] == True

def test_predict_status():
    """ voir si la route /predict repond sans planter"""
    if MODE == "nlp":
        payload = {"text": CONFIG["nlp"]["texte_neutre"]}
    else:
        payload = CONFIG[MODE]["sample_input"]

    r = client.post("/predict", json=payload)
    assert r.status_code == 200

def test_predict_structure():
    """La reponse contient les bons champs"""
    if MODE == "nlp":
        payload = {"text": CONFIG["nlp"]["texte_neutre"]}
        r = client.post("/predict", json=payload)
        data = r.json()
        assert "sentiment"  in data
        assert "confiance"  in data
        assert "texte_recu" in data

    elif MODE == "classification":
        payload = CONFIG["classification"]["sample_input"]
        r = client.post("/predict", json=payload)
        data = r.json()
        assert "prediction" in data
        assert "confiance"  in data
        assert "type"       in data

    elif MODE == "regression":
        payload = CONFIG["regression"]["sample_input"]
        r = client.post("/predict", json=payload)
        data = r.json()
        assert "prediction" in data
        assert "type"       in data
        assert "confiance"  not in data  # pas de confiance en régression

def test_mauvaise_requete():
    """Requete vide retourne une erreur propre"""
    r = client.post("/predict", json={})
    assert r.status_code == 422  #donnees manquantes

# ══════════════════════════════════════════════
#   TESTS NLP
# ══════════════════════════════════════════════

@pytest.mark.skipif(MODE != "nlp", reason="NLP uniquement")
def test_nlp_sentiment_positif():
    """Texte positif predit positive"""
    r = client.post("/predict",
        json={"text": CONFIG["nlp"]["texte_positif"]})
    assert r.json()["sentiment"] == "positive"
    assert r.json()["confiance"] > CONFIG["nlp"]["seuil_confiance"]

@pytest.mark.skipif(MODE != "nlp", reason="NLP uniquement")
def test_nlp_sentiment_negatif():
    """Texte negatif predit negative"""
    r = client.post("/predict",
        json={"text": CONFIG["nlp"]["texte_negatif"]})
    assert r.json()["sentiment"] == "negative"
    assert r.json()["confiance"] > CONFIG["nlp"]["seuil_confiance"]

@pytest.mark.skipif(MODE != "nlp", reason="NLP uniquement")
def test_nlp_confiance_valide():
    """Confiance toujours entre 0 et 1"""
    r = client.post("/predict",
        json={"text": CONFIG["nlp"]["texte_neutre"]})
    assert 0.0 <= r.json()["confiance"] <= 1.0

@pytest.mark.skipif(MODE != "nlp", reason="NLP uniquement")
def test_nlp_predict_batch():
    """predict_batch retourne autant de resultats que d'entrees"""
    textes = CONFIG["nlp"]["batch_textes"]
    r = client.post("/predict_batch", json={"texts": textes})
    assert r.status_code == 200
    assert r.json()["total"] == len(textes)
    assert len(r.json()["predictions"]) == len(textes)

@pytest.mark.skipif(MODE != "nlp", reason="NLP uniquement")
def test_nlp_sentiment_valide():
    """Sentiment est toujours positive ou negative"""
    r = client.post("/predict",
        json={"text": CONFIG["nlp"]["texte_positif"]})
    assert r.json()["sentiment"] in ["positive", "negative"]

# ══════════════════════════════════════════════
#   TESTS CLASSIFICATION
# ══════════════════════════════════════════════

@pytest.mark.skipif(MODE != "classification", reason="Classification uniquement")
def test_classif_prediction_valide():
    """Prediction est une classe valide"""
    r = client.post("/predict",
        json=CONFIG["classification"]["sample_input"])
    assert r.json()["prediction"] in CONFIG["classification"]["classes_valides"]

@pytest.mark.skipif(MODE != "classification", reason="Classification uniquement")
def test_classif_confiance_valide():
    """Confiance entre 0 et 1"""
    r = client.post("/predict",
        json=CONFIG["classification"]["sample_input"])
    assert 0.0 <= r.json()["confiance"] <= 1.0

@pytest.mark.skipif(MODE != "classification", reason="Classification uniquement")
def test_classif_type_correct():
    """Type retourne classification"""
    r = client.post("/predict",
        json=CONFIG["classification"]["sample_input"])
    assert r.json()["type"] == "classification"

# ══════════════════════════════════════════════
#   TESTS REGRESSION
# ══════════════════════════════════════════════

@pytest.mark.skipif(MODE != "regression", reason="Regression uniquement")
def test_regression_prediction_positive():
    """Prediction est un nombre positif"""
    r = client.post("/predict",
        json=CONFIG["regression"]["sample_input"])
    assert r.json()["prediction"] > CONFIG["regression"]["valeur_min"]

@pytest.mark.skipif(MODE != "regression", reason="Regression uniquement")
def test_regression_prediction_numerique():
    """Prediction est un nombre"""
    r = client.post("/predict",
        json=CONFIG["regression"]["sample_input"])
    assert isinstance(r.json()["prediction"], (int, float))

@pytest.mark.skipif(MODE != "regression", reason="Regression uniquement")
def test_regression_type_correct():
    """Type retourne regression"""
    r = client.post("/predict",
        json=CONFIG["regression"]["sample_input"])
    assert r.json()["type"] == "regression"





    """CONFIG = {
    "mode": "classification",
    "classification": {
        "sample_input": {
            "pclass": 1, "sex": "female",
            "age": 25.0, "fare": 100.0
        },
        "classes_valides": ["0", "1"],
        "seuil_confiance": 0.50,
    }
}

CONFIG = {
    "mode": "regression",
    "regression": {
        "sample_input": {
            "OverallQual": 7,
            "GrLivArea": 1500
        },
        "valeur_min": 0,
    }
}
"""