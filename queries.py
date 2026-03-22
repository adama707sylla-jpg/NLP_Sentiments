import pandas as pd
from sqlalchemy import create_engine, text, inspect
from config import DATABASE_URL

engine = create_engine(DATABASE_URL)


def get_data_ml(table, target=None, drop_cols=None, dropna=True):
    query = f"SELECT * FROM {table}"
    df = pd.read_sql(query, engine)

    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    if dropna:
        avant = len(df)
        df = df.dropna()
        print(f"⚠️  {avant - len(df)} lignes supprimées (valeurs manquantes)")

    if target and target in df.columns:
        X = df.drop(columns=[target])
        y = df[target]
        print(f"✅ {len(df)} lignes | {len(X.columns)} features | cible : {target}")
        return X, y

    print(f"✅ {len(df)} lignes chargées depuis '{table}'")
    return df


def get_data_quality(table):
    inspector = inspect(engine)
    colonnes  = [col["name"] for col in inspector.get_columns(table)]

    total = pd.read_sql(f"SELECT COUNT(*) AS total FROM {table}", engine)["total"][0]

    print(f"\n📊 Rapport qualité — table : '{table}'")
    print(f"   Total lignes : {total}")
    print(f"\n{'Colonne':<25} {'Manquants':>10} {'%':>8} {'Statut':>12}")
    print("-" * 60)

    rapport = []
    for col in colonnes:
        query  = f'SELECT COUNT(*) - COUNT("{col}") AS manquants FROM {table}'
        manq   = pd.read_sql(query, engine)["manquants"][0]
        pct    = round(manq / total * 100, 1) if total > 0 else 0
        statut = "🔴 critique" if pct > 60 else "⚠️  attention" if pct > 20 else "✅ ok"
        print(f"{col:<25} {manq:>10} {pct:>7}% {statut:>12}")
        rapport.append({"colonne": col, "manquants": manq, "pct": pct})

    return pd.DataFrame(rapport)


def get_stats_groupe(table, groupe, cible=None):
    if cible:
        query = f"""
            SELECT
                "{groupe}",
                COUNT(*)                                AS total,
                ROUND(AVG("{cible}")::numeric, 2)       AS moyenne,
                ROUND(MIN("{cible}")::numeric, 2)       AS min,
                ROUND(MAX("{cible}")::numeric, 2)       AS max
            FROM {table}
            GROUP BY "{groupe}"
            ORDER BY "{groupe}"
        """
    else:
        query = f"""
            SELECT "{groupe}", COUNT(*) AS total
            FROM {table}
            GROUP BY "{groupe}"
            ORDER BY total DESC
        """

    df = pd.read_sql(query, engine)
    print(f"\n📊 Stats '{table}' groupé par '{groupe}' :")
    print(df.to_string(index=False))
    return df


def get_anomalies(table, regles=None):
    if not regles:
        print("⚠️  Aucune règle définie — retour table complète")
        return pd.read_sql(f"SELECT * FROM {table} LIMIT 10", engine)

    conditions = " OR ".join(regles)
    query      = f"SELECT * FROM {table} WHERE {conditions}"
    df         = pd.read_sql(query, engine)

    print(f"⚠️  {len(df)} anomalie(s) dans '{table}'")
    if len(df) > 0:
        print(df.head())
    return df


def get_outliers_iqr(table, colonne):
    query = f"""
        WITH stats AS (
            SELECT
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY "{colonne}") AS q1,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY "{colonne}") AS q3
            FROM {table}
        )
        SELECT t.*
        FROM {table} t, stats s
        WHERE t."{colonne}" > s.q3 + 1.5 * (s.q3 - s.q1)
           OR t."{colonne}" < s.q1 - 1.5 * (s.q3 - s.q1)
        ORDER BY t."{colonne}" DESC
    """
    df = pd.read_sql(query, engine)
    print(f'💰 {len(df)} outlier(s) sur "{colonne}" dans {table}')
    return df


def run_query(sql):
    df = pd.read_sql(sql, engine)
    print(f"✅ {len(df)} lignes retournées")
    return df


# ══════════════════════════════════════════════
#   MAIN — test projet immobilier
# ══════════════════════════════════════════════

if __name__ == "__main__":
    print("\n=== TEST QUERIES.PY — Projet NLP ===\n")

    # 1. Charger données ML
    X, y = get_data_ml(
        table     = "avis",
        target    = "sentiment",
        drop_cols = ["titre"]
    )

    # 2. Qualité des données
    get_data_quality("avis")

    # 3. Stats par groupe
    get_stats_groupe("avis", groupe="sentiment", cible="sentiment")

    # 4. Anomalies
    #get_anomalies("avis", regles=[
        #'"sentiment" < 0',
        #'"" < 0'
    #])

    # 5. Outliers
    #get_outliers_iqr("maisons", "SalePrice")
    #get_outliers_iqr("maisons", "GrLivArea")

    # 6. Requête libre
    df = run_query("""
        SELECT sentiment, COUNT(*) as total
        FROM avis
        GROUP BY sentiment
        ORDER BY sentiment
    """)
    print(df)

    print("\n✅ queries.py OK — prêt pour le projet nlp !")