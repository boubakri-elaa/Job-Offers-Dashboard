import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# ==== chemins ====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CLUSTERS_PATH = os.path.join(BASE_DIR, "data", "processed", "offres_clusters.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)
ML_PATH = os.path.join(PROCESSED_DIR, "offres_ml.csv")

# 1) Charger les données avec clusters
df = pd.read_csv(CLUSTERS_PATH, encoding="utf-8")

# 2) Créer une cible binaire à partir du cluster
#    Ici : cluster 3 = "high_demand" (1), les autres = 0
df["high_demand"] = (df["cluster_metier"] == 3).astype(int)

X = df["texte_complet"].fillna("")
y = df["high_demand"]

# 3) Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4) Pipeline TF-IDF + Régression Logistique
clf = Pipeline(
    steps=[
        ("tfidf", TfidfVectorizer(max_features=2000)),
        ("logreg", LogisticRegression(max_iter=1000)),
    ]
)

# 5) Entraînement
clf.fit(X_train, y_train)

# 6) Évaluation simple
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy (test) : {acc:.3f}")

# 7) Prédictions sur toutes les offres
df["pred_high_demand"] = clf.predict(X)

# 8) Sauvegarde
df.to_csv(ML_PATH, index=False, encoding="utf-8")
print("✅ Fichier enrichi (ML) sauvegardé dans :", ML_PATH)
print(df[["Titre", "cluster_metier", "high_demand", "pred_high_demand"]].head())
