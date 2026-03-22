# monitoring.py — Projet NLP Sentiments
import pandas as pd
import numpy as np
import joblib
import re
from config import MODEL_PATH, REFERENCE_PATH, CURRENT_PATH, DRIFT_SEUIL

# Charger le modèle
model = joblib.load(MODEL_PATH)

# Fonction nettoyage
def nettoyer_texte(texte):
    texte = str(texte).lower()
    texte = re.sub(r'http\S+', '', texte)
    texte = re.sub(r'@\w+',    '', texte)
    texte = re.sub(r'#\w+',    '', texte)
    texte = re.sub(r'[^a-z\s]', ' ', texte)
    texte = re.sub(r'\s+',     ' ', texte).strip()
    return texte


#  Distribution des prédictions
def monitorer_distribution(predictions):
    """
    Alerte si la distribution des prédictions dérive.
    Normalement : ~50% positif, ~50% négatif (dataset équilibré)
    Si en prod : 95% positif → suspect !
    """
    print("\n Distribution des prédictions :")
    dist = pd.Series(predictions).value_counts(normalize=True)
    print(dist)

    pct_positif = dist.get("positive", 0) * 100
    pct_negatif = dist.get("negative", 0) * 100

    print(f"\n   Positif : {pct_positif:.1f}%")
    print(f"   Négatif : {pct_negatif:.1f}%")

    if pct_positif > 90:
        print(" ALERTE : +90% predictions positives — suspect !")
    elif pct_negatif > 90:
        print(" ALERTE : +90% predictions negatives — suspect !")
    elif pct_positif > 70:
        print(" ATTENTION : distribution desequilibree")
    else:
        print(" Distribution normale")

    return dist



#   FONCTION 2 — Longueur des textes

def monitorer_longueur(df_ref, df_curr, col_texte="texte"):
    """
    Alerte si la longueur moyenne des textes dérive.
    Textes très courts en prod → spam ou bot ?
    Textes très longs → changement de source de données ?
    """
    print("\n Longueur des textes :")

    long_ref  = df_ref[col_texte].str.len().mean()
    long_curr = df_curr[col_texte].str.len().mean()
    diff      = abs(long_curr - long_ref)
    pct       = round(diff / long_ref * 100, 1) if long_ref > 0 else 0

    print(f"   Reference    : {long_ref:.0f} caractères en moyenne")
    print(f"   Production   : {long_curr:.0f} caractères en moyenne")
    print(f"   Différence   : {diff:.0f} chars ({pct}%)")

    if pct > DRIFT_SEUIL:
        print(f"ALERTE : longueur dérive de {pct}% > seuil {DRIFT_SEUIL}%")
    else:
        print(f" Longueur stable (écart {pct}% < seuil {DRIFT_SEUIL}%)")

    return {"ref": long_ref, "curr": long_curr, "diff_pct": pct}


# ══════════════════════════════════════════════
#   FONCTION 3 — Vocabulaire inconnu
# ══════════════════════════════════════════════
def monitorer_vocab_inconnu(df_curr, col_texte="texte"):
    """
    Détecte les mots en production absents du vocabulaire d'entraînement.
    Si beaucoup de mots inconnus → le modèle prédit dans le vide !
    """
    print("\n Vocabulaire inconnu :")

    # Récupérer le vocabulaire appris par TF-IDF
    vocab_train = set(model.named_steps["tfidf"].vocabulary_.keys())

    # Récupérer tous les mots de production
    textes_clean = df_curr[col_texte].apply(nettoyer_texte)
    tous_mots    = set(" ".join(textes_clean).split())

    # Trouver les mots inconnus
    mots_inconnus = tous_mots - vocab_train
    pct_inconnus  = round(len(mots_inconnus) / len(tous_mots) * 100, 1)

    print(f"   Vocabulaire train  : {len(vocab_train):,} mots")
    print(f"   Mots en production : {len(tous_mots):,} mots")
    print(f"   Mots inconnus      : {len(mots_inconnus):,} ({pct_inconnus}%)")

    if pct_inconnus > 50:
        print(f"ALERTE : {pct_inconnus}% mots inconnus — reentainer le modele !")
    elif pct_inconnus > 35:
        print(f" ATTENTION : {pct_inconnus}% mots inconnus")
    else:
        print(f" Vocabulaire stable")

    # Afficher quelques mots inconnus
    if mots_inconnus:
        exemples = list(mots_inconnus)[:10]
        print(f"   Exemples : {exemples}")

    return {"total": len(tous_mots), "inconnus": len(mots_inconnus), "pct": pct_inconnus}


# ══════════════════════════════════════════════
#   FONCTION 4 — Confiance moyenne
# ══════════════════════════════════════════════
def monitorer_confiance(df_curr, col_texte="texte"):
    """
    Surveille la confiance moyenne des prédictions.
    Si confiance baisse → le modèle devient incertain
    → données trop différentes de l'entraînement
    """
    print("\n🎯 Confiance des prédictions :")

    textes_clean = df_curr[col_texte].apply(nettoyer_texte).tolist()
    probas       = model.predict_proba(textes_clean).max(axis=1)

    confiance_moy = round(probas.mean(), 4)
    confiance_min = round(probas.min(), 4)
    pct_incertains = round((probas < 0.70).mean() * 100, 1)

    print(f"   Confiance moyenne  : {confiance_moy:.2%}")
    print(f"   Confiance minimum  : {confiance_min:.2%}")
    print(f"   Prédictions < 70%  : {pct_incertains}%")

    if confiance_moy < 0.70:
        print(" ALERTE : confiance moyenne trop basse — modèle incertain !")
    elif confiance_moy < 0.80:
        print("⚠️  ATTENTION : confiance en baisse")
    else:
        print("✅ Confiance normale")

    return {
        "moyenne"    : confiance_moy,
        "minimum"    : confiance_min,
        "pct_bas"    : pct_incertains
    }


#   RAPPORT COMPLET

def rapport_complet(df_ref, df_curr, col_texte="texte"):
    """Lance tous les contrôles en une seule fois."""

    print("\n" + "="*55)
    print("   RAPPORT MONITORING — NLP Sentiments")
    print("="*55)
    print(f"   Reference  : {len(df_ref)} textes")
    print(f"   Production : {len(df_curr)} textes")

    # Prédire sur les données de production
    textes_clean = df_curr[col_texte].apply(nettoyer_texte).tolist()
    predictions  = model.predict(textes_clean)

    # Lancer tous les contrôles
    monitorer_distribution(predictions)
    monitorer_longueur(df_ref, df_curr, col_texte)
    monitorer_vocab_inconnu(df_curr, col_texte)
    monitorer_confiance(df_curr, col_texte)

    print("\n" + "="*55)
    print("   MONITORING TERMINE")
    print("="*55)



#   MAIN
if __name__ == "__main__":

    # Charger les données de référence (entraînement)
    print("Chargement des données...")
    df_ref  = pd.read_csv(
        REFERENCE_PATH,
        header=None,
        names=["sentiment", "titre", "texte"],
        encoding="latin-1"
    ).sample(1000, random_state=42)

    # Simuler des données de production
    # (en réalité ce seraient les nouveaux avis reçus)
    df_curr = pd.read_csv(
        REFERENCE_PATH,
        header=None,
        names=["sentiment", "titre", "texte"],
        encoding="latin-1"
    ).sample(1000, random_state=99)   # ← random_state différent

    # Lancer le rapport complet
    rapport_complet(df_ref, df_curr, col_texte="texte")
