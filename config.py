# config.py
import os

# ── Base de données ──────────────────────────
DB_USER     = os.getenv("DB_USER",     "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "1234")
DB_HOST     = os.getenv("DB_HOST",     "localhost")
DB_PORT     = os.getenv("DB_PORT",     "5432")
DB_NAME     = os.getenv("DB_NAME",     "nlp_db")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ── Modèle ───────────────────────────────────
MODEL_PATH   = os.getenv("MODEL_PATH", "modele.pkl")
PROJECT_NAME = "Analyser_Sentiments_NLP"

# ── MLflow ───────────────────────────────────
MLFLOW_URI = os.getenv("MLFLOW_URI", "sqlite:///mlflow.db")

# ── Monitoring ───────────────────────────────
REFERENCE_PATH = os.getenv("REFERENCE_PATH", "train.csv")
CURRENT_PATH   = os.getenv("CURRENT_PATH",   "test.csv")
DRIFT_SEUIL    = 20.0

