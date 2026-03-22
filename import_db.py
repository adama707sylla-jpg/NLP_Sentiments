import pandas as pd
from sqlalchemy import create_engine
from config import DATABASE_URL

engine = create_engine(DATABASE_URL)

# Charger le CSV brut — TOUTES les colonnes
df = pd.read_csv("train.csv",
                 header=None,
                 names=["sentiment", "titre", "texte"],
                 encoding="latin-1")

df = df.sample(50000, random_state=42)

print(f"Shape : {df.shape}")
print(f"Colonnes : {list(df.columns)}")
print(f"sentiments : {df['sentiment'].value_counts().to_dict()}")

# Importer TOUT dans PostgreSQL
df.to_sql("avis", engine, if_exists="replace", index=False)

print(f"Table 'avis' créée avec {len(df)} lignes !")
