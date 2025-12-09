import os
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# ==== chemins ====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
INTERIM_PATH = os.path.join(BASE_DIR, "data", "interim", "offres_hellowork_clean.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)
CLUSTERS_PATH = os.path.join(PROCESSED_DIR, "offres_clusters.csv")

# 1) Charger les données nettoyées
df = pd.read_csv(INTERIM_PATH, encoding="utf-8")

# 2) Vectoriser le texte (TF-IDF sur texte_complet)
french_stopwords = ["le", "la", "les", "de", "des", "du", "un", "une", "et", "en", "pour", "avec", "sur", "dans"]

vectorizer = TfidfVectorizer(
    max_features=2000,
    stop_words=french_stopwords,
)

X_text = vectorizer.fit_transform(df["texte_complet"].fillna(""))

# 3) Clustering KMeans
n_clusters = 5  # tu peux tester d'autres valeurs
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
clusters = kmeans.fit_predict(X_text)

# 4) Ajouter les clusters au DataFrame
df["cluster_metier"] = clusters

# 5) Sauvegarder pour le dashboard
df.to_csv(CLUSTERS_PATH, index=False, encoding="utf-8")

print("✅ Clustering terminé.")
print("Fichier enrichi sauvegardé dans :", CLUSTERS_PATH)
print(df[["Titre", "cluster_metier"]].head())
print(df["cluster_metier"].value_counts())
