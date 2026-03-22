import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from config import MODEL_PATH, PROJECT_NAME, MLFLOW_URI
from queries import get_data_ml

import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings("ignore")


#   CONFIG
CONFIG = {
    "table"        : "avis",
    "target"       : "sentiment",
    "text_col"     : "texte",           
    "drop_cols"    : ["titre"],
    "mode"         : "classification",
    "test_size"    : 0.2,
    "random_state" : 42,
    "params"       : {"max_iter": 1000}
}

#   FONCTION NETTOYAGE
import re

def nettoyer_texte(texte):
    texte = str(texte).lower()
    texte = re.sub(r'http\S+', '', texte)
    texte = re.sub(r'@\w+', '', texte)
    texte = re.sub(r'#\w+', '', texte)
    texte = re.sub(r'[^a-z\s]', ' ', texte)
    texte = re.sub(r'\s+', ' ', texte).strip()
    return texte


#   PIPELINE PRINCIPAL
def run_training(config=CONFIG):

    print("\n" + "="*50)
    print(f"  {PROJECT_NAME}  [NLP]")
    print("="*50)

    #Chargement PostgreSQL
    print("\n Chargement des donnees...")
    df = get_data_ml(
        table     = config["table"],
        target    = None,              # on charge tout d'abord
        drop_cols = config["drop_cols"],
        dropna    = False
    )

    #Nettoyage texte
    print("\n Nettoyage du texte...")
    df[config["text_col"]] = df[config["text_col"]].apply(nettoyer_texte)
    df = df.dropna(subset=[config["text_col"], config["target"]])
    print(f"   {len(df)} lignes apres nettoyage")

    # Convertir sentiment en label lisible
    df[config["target"]] = df[config["target"]].map(
        {1: "negative", 2: "positive"}
    )

    # Split train/test 
    print("\n Split train/test...")
    X = df[config["text_col"]]   
    y = df[config["target"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = config["test_size"],
        random_state = config["random_state"]
    )
    print(f"   Train : {len(X_train)} | Test : {len(X_test)}")

    # Pipeline NLP 
    print("\n  Construction pipeline NLP...")
    modele_v2 = Pipeline([
        ("tfidf",  TfidfVectorizer(
            max_features = 10000,
            ngram_range  = (1, 2)
        )),
        ("modele", LogisticRegression(**config["params"]))
    ])

    modele_v2.fit(X_train, y_train)
    y_pred = modele_v2.predict(X_test)

    # MLflow tracking 
    print("\n Tracking MLflow...")
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(PROJECT_NAME)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, pos_label="positive")

    print(f"\n----Evaluation LogisticRegression----")
    print(f" Accuracy : {acc*100:.2f}%")
    print(f" F1 Score : {f1*100:.2f}%")
    print("-" * 35)

    with mlflow.start_run(run_name="LogReg_NLP_v3"):
        mlflow.log_param("model",        "LogisticRegression")
        mlflow.log_param("max_features", 5000)
        mlflow.log_param("ngram_range",  "(1,2)")
        mlflow.log_param("table",        config["table"])
        mlflow.log_params(config["params"])
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1",       f1)
        mlflow.sklearn.log_model(modele_v3, artifact_path="model")
        print("   Experience enregistree dans MLflow !")

    # Sauvegarde 
    print(f"\n Sauvegarde → {MODEL_PATH}")
    joblib.dump(modele_v2, MODEL_PATH)

    print("\n" + "="*50)
    print("  TRAINING TERMINE !")
    print("="*50 + "\n")

    return modele_v2


if __name__ == "__main__":
    run_training()