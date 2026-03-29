FROM python:3.11-slim

WORKDIR /app

# 1. Installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copier les fichiers nécessaires
COPY modele.pkl .
COPY app.py .
COPY config.py .

# 3. Lancer l'API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# uvicorn : serveur web
# app:app  | fichier app.py, object app
# 0.0.0.0: ecoute toutes les connexions 
# --port 10000: port impose par render
#rsquisitoryUri:718100329950.dkr.ecr.eu-west-3.amazonaws.com/nlp-sentiments