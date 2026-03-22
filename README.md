# 🧠 NLP Sentiments — Analyse de Sentiments Amazon

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-latest-green)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-latest-orange)
![MLflow](https://img.shields.io/badge/MLflow-tracking-blue)
![Docker](https://img.shields.io/badge/Docker-containerized-blue)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-green)
![Render](https://img.shields.io/badge/Deploy-Render-purple)

> Pipeline MLOps complet pour la classification de sentiments d'avis clients Amazon.  
> De la base de données PostgreSQL au déploiement en production sur Render.

---

## 🎯 Objectif

Classifier automatiquement les avis clients Amazon en **positif** ou **négatif**  
pour permettre au service client de prioriser les réclamations.

| Métrique | Résultat |
|----------|----------|
| Accuracy | **88.5%** |
| F1 Score | **88.4%** |
| Dataset  | 50 000 avis Amazon |
| Modèle   | LogisticRegression + TF-IDF |

---

## 🏗️ Architecture

```
train.csv
    ↓
import_db.py  →  PostgreSQL (nlp_db)
    ↓
queries.py    →  Extraction des données
    ↓
train.py      →  TF-IDF + LogisticRegression + MLflow
    ↓
modele.pkl    →  Modèle sauvegardé
    ↓
app.py        →  API FastAPI
    ↓
Docker        →  Containerisation
    ↓
Render        →  Production
```

---

## 📁 Structure du projet

```
nlp_sentiments/
├── .github/
│   └── workflows/
│       └── ci.yml          # CI/CD GitHub Actions
├── notebooks/
│   └── nlp.ipynb           # EDA + comparaison modèles
├── config.py               # Variables centralisées (os.getenv)
├── import_db.py            # CSV → PostgreSQL
├── queries.py              # Fonctions génériques PostgreSQL
├── train.py                # Pipeline ML complet
├── app.py                  # API FastAPI
├── monitoring.py           # Détection data drift
├── test_api.py             # Tests automatiques pytest
├── Dockerfile              # Containerisation
├── requirements.txt        # Dépendances
└── modele.pkl              # Modèle entraîné
```

---

## 🚀 Démarrage rapide

### Prérequis

```bash
Python 3.11+
PostgreSQL 15+
Docker (optionnel)
```

### Installation

```bash
# Cloner le repo
git clone https://github.com/adama707sylla-jpg/NLP_Sentiments.git
cd NLP_Sentiments

# Installer les dépendances
pip install -r requirements.txt
```

### Configuration

```bash
# Variables d'environnement (ou modifier config.py)
export DB_USER=postgres
export DB_PASSWORD=1234
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=nlp_db
```

### Lancer le pipeline complet

```bash
# 1. Créer la base PostgreSQL
psql -U postgres -c "CREATE DATABASE nlp_db;"

# 2. Importer les données (train.csv requis)
python import_db.py

# 3. Entraîner le modèle
python train.py

# 4. Lancer l'API
uvicorn app:app --reload

# 5. Lancer les tests
pytest test_api.py -v
```

---

## 🌐 API — Endpoints

### Base URL (Production)
```
https://nlp-sentiments-api.onrender.com
```

### GET /
```json
{
  "status": "ok",
  "projet": "Analyser_Sentiments_NLP",
  "modele": "LogisticRegression + TF-IDF"
}
```

### POST /predict
**Prédire le sentiment d'un texte**

```bash
curl -X POST https://nlp-sentiments-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is absolutely amazing!"}'
```

```json
{
  "sentiment": "positive",
  "confiance": 0.9292,
  "texte_recu": "This product is absolutely amazing!"
}
```

### POST /predict_batch
**Prédire plusieurs textes en une seule requête**

```bash
curl -X POST https://nlp-sentiments-api.onrender.com/predict_batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Amazing product!", "Terrible quality!", "It was okay"]}'
```

```json
{
  "total": 3,
  "predictions": [
    {"texte": "Amazing product!", "sentiment": "positive", "confiance": 0.94},
    {"texte": "Terrible quality!", "sentiment": "negative", "confiance": 0.91},
    {"texte": "It was okay", "sentiment": "positive", "confiance": 0.61}
  ]
}
```

### GET /health
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### 📖 Documentation interactive
```
https://nlp-sentiments-api.onrender.com/docs
```

---

## 🤖 Modèle ML

### Pipeline NLP

```python
Pipeline([
    ('tfidf',  TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
    ('modele', LogisticRegression(max_iter=1000))
])
```

### Comparaison des modèles (MLflow)

| Modèle | Accuracy | F1 Score |
|--------|----------|----------|
| **LogisticRegression** | **88.5%** | **88.4%** |
| LinearSVC | 87.1% | 87.1% |
| MultinomialNB | 84.6% | 84.6% |
| RandomForest | 80.2% | 80.7% |

### Comparaison des paramètres TF-IDF

| Run | max_features | ngram_range | Accuracy |
|-----|-------------|-------------|----------|
| v1  | 5 000 | (1,2) | 87.85% |
| v2  | 10 000 | (1,2) | **88.48%** ✅ |
| v3  | 5 000 | (1,3) | 88.00% |

---

## 🐳 Docker

```bash
# Builder l'image
docker build -t nlp-api .

# Lancer le conteneur
docker run -p 8000:8000 nlp-api

# Tester
curl http://localhost:8000/health
```

---

## ✅ Tests

```bash
pytest test_api.py -v
```

```
test_root                    PASSED ✅
test_health                  PASSED ✅
test_predict_status          PASSED ✅
test_predict_structure       PASSED ✅
test_mauvaise_requete        PASSED ✅
test_nlp_sentiment_positif   PASSED ✅
test_nlp_sentiment_negatif   PASSED ✅
test_nlp_confiance_valide    PASSED ✅
test_nlp_predict_batch       PASSED ✅
test_nlp_sentiment_valide    PASSED ✅

10 passed in 3.15s
```

---

## 📊 Monitoring

```bash
python monitoring.py
```

Surveille automatiquement :
- **Distribution** des prédictions (alerte si 90%+ dans une classe)
- **Longueur** des textes en production vs référence
- **Vocabulaire inconnu** (mots absents du vocabulaire TF-IDF)
- **Confiance** moyenne des prédictions (alerte si < 70%)

---

## 🔄 CI/CD GitHub Actions

```
git push
    ↓
✅ pytest 10/10 tests
    ↓
✅ docker build
    ↓
✅ deploy Render
```

---

## 🛠️ Stack technique

| Composant | Technologie |
|-----------|-------------|
| Langage | Python 3.11 |
| API | FastAPI + Uvicorn |
| ML | Scikit-learn |
| NLP | TF-IDF + LogisticRegression |
| Base de données | PostgreSQL + SQLAlchemy |
| Experiment tracking | MLflow |
| Tests | pytest |
| Containerisation | Docker |
| CI/CD | GitHub Actions |
| Déploiement | Render |

---

## 📈 Résultats

```
Dataset    : 50 000 avis Amazon (équilibré)
             25 039 positifs / 24 961 négatifs

Modèle     : LogisticRegression + TF-IDF
Accuracy   : 88.5%
F1 Score   : 88.4%

Exemple :
"This product is amazing!" → positive (93%)
"Terrible waste of money!" → negative (91%)
```

---

## 👤 Auteur

**Adama Sylla**  
MLOps Junior  
[GitHub](https://github.com/adama707sylla-jpg)

---

## 📄 Projets connexes

| Projet | Type | Performance |
|--------|------|-------------|
| [Titanic MLOps](https://github.com/adama707sylla-jpg/titan) | Classification | Accuracy 82.77% |
| [Immobilier MLOps](https://github.com/adama707sylla-jpg/immo) | Régression | R2 = 0.90 |
| **NLP Sentiments** | **NLP** | **Accuracy 88.5%** |
