import os
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.sparse import hstack


# ========================================
# CHEMINS
# ========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
INTERIM_PATH = os.path.join(BASE_DIR, "data", "interim", "offres_hellowork_clean.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)
CLUSTERS_PATH = os.path.join(PROCESSED_DIR, "offres_clusters.csv")


# ========================================
# CHARGEMENT DES DONNÉES
# ========================================
print("📂 Chargement des données nettoyées...")
df = pd.read_csv(INTERIM_PATH, encoding="utf-8")
print(f"✅ {len(df)} offres chargées")


# ========================================
# PRÉPARATION TEXTE POUR TF-IDF
# ========================================
def nettoyer_texte(texte: str) -> str:
    t = str(texte).lower()
    remplacements = [
        "h/f", "h / f", "(h/f)", "(h / f)",
        " cdi ", " cdd ", " stage ", " alternance ",
        " france ", " hf ", " h f "
    ]
    for r in remplacements:
        t = t.replace(r, " ")
    return t

df["texte_tf"] = df["texte_complet"].fillna("").apply(nettoyer_texte)


# ========================================
# VECTORISATION TF-IDF
# ========================================
print("\n🔢 Vectorisation du texte (TF-IDF)...")

french_stopwords = [
    "le", "la", "les", "de", "des", "du", "un", "une", "et",
    "en", "pour", "avec", "sur", "dans", "par", "au", "aux",
    "d'", "l'", "h/f", "hf", "offre", "poste"
]

vectorizer = TfidfVectorizer(
    max_features=1000,
    stop_words=french_stopwords,
    min_df=2,
    max_df=0.7,
)

X_text = vectorizer.fit_transform(df["texte_tf"])
print(f"✅ Matrice TF-IDF: {X_text.shape[0]} offres × {X_text.shape[1]} mots-clés")


# ========================================
# AJOUT DE DOMAINE_METIER COMME FEATURE
# ========================================
print("\n➕ Ajout de 'Domaine_metier' comme features (one-hot)...")

domain_dummies = pd.get_dummies(df["Domaine_metier"], sparse=True)
X_full = hstack([X_text, domain_dummies.values])

print(f"✅ Matrice finale: {X_full.shape[0]} offres × {X_full.shape[1]} features")


# ========================================
# TEST DES 3 ALGORITHMES
# ========================================
print("\n🧪 Test de KMeans, Agglomerative, DBSCAN...")

n_clusters = 8
results = {}
labels_dict = {}

# 1) KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
labels_kmeans = kmeans.fit_predict(X_full)
score_kmeans = silhouette_score(X_full, labels_kmeans)
results["KMeans"] = {"labels": labels_kmeans, "score": score_kmeans, "k": n_clusters}
labels_dict["KMeans"] = labels_kmeans

# 2) Agglomerative
agg = AgglomerativeClustering(n_clusters=n_clusters)
labels_agg = agg.fit_predict(X_full.toarray())
score_agg = silhouette_score(X_full, labels_agg)
results["Agglomerative"] = {"labels": labels_agg, "score": score_agg, "k": n_clusters}
labels_dict["Agglomerative"] = labels_agg

# 3) DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5, n_jobs=-1)
labels_db = dbscan.fit_predict(X_full)
if len(set(labels_db)) > 1:
    score_db = silhouette_score(X_full, labels_db)
else:
    score_db = -1.0
results["DBSCAN"] = {"labels": labels_db, "score": score_db, "k": len(set(labels_db))}
labels_dict["DBSCAN"] = labels_db

print("\n📊 Scores de clustering (silhouette) avec Domaine_metier:")
for name, info in results.items():
    print(f" - {name} (k≈{info['k']}) : {info['score']:.3f}")


print("\nℹ️ Choisis l'algorithme que tu veux utiliser parmi: KMeans, Agglomerative, DBSCAN")
# ====== CHOIX MANUEL ======
chosen_algo = "KMeans"  # <=== change ici si tu veux tester un autre
# ==========================

print(f"\n✅ Algorithme choisi manuellement : {chosen_algo}")
df["cluster_id"] = labels_dict[chosen_algo]


# ========================================
# NOMMER LES CLUSTERS (via Domaine_metier)
# ========================================
print("\n🏷️ Attribution des noms aux clusters...")

def nommer_cluster(cluster_id: int, df_cluster: pd.DataFrame) -> str:
    domaines = df_cluster["Domaine_metier"].value_counts()

    if len(domaines) == 0:
        return f"Cluster {cluster_id}"

    domaine_principal = domaines.index[0]

    if len(domaines) > 1 and domaines.iloc[1] > len(df_cluster) * 0.3:
        domaine_secondaire = domaines.index[1]
        return f"{domaine_principal}/{domaine_secondaire}"

    return domaine_principal


cluster_names = {}
for cluster_id in sorted(df["cluster_id"].unique()):
    df_cluster = df[df["cluster_id"] == cluster_id]
    cluster_names[cluster_id] = nommer_cluster(cluster_id, df_cluster)

df["cluster_nom"] = df["cluster_id"].map(cluster_names)


# ========================================
# AFFICHAGE DES RÉSULTATS
# ========================================
print("\n📊 Résultats du clustering final (algo choisi):")
print("=" * 60)

for cluster_id in sorted(df["cluster_id"].unique()):
    df_cluster = df[df["cluster_id"] == cluster_id]
    print(f"\n🔹 Cluster {cluster_id}: {cluster_names[cluster_id]}")
    print(f"   Nombre d'offres: {len(df_cluster)}")

    top_titres = df_cluster["Titre"].value_counts().head(3)
    print("   Top 3 métiers:")
    for titre, count in top_titres.items():
        print(f"      • {titre} ({count})")

print("\n📈 Répartition domaines × clusters:")
print(df.groupby(["Domaine_metier", "cluster_nom"])["Titre"].count())


# ========================================
# SAUVEGARDE
# ========================================
df.to_csv(CLUSTERS_PATH, index=False, encoding="utf-8")
print(f"\n✅ Fichier enrichi avec clusters sauvegardé dans: {CLUSTERS_PATH}")
print(f"✅ Colonnes ajoutées: 'cluster_id', 'cluster_nom'")
print("\n👀 Aperçu des données avec clusters:")
print(df[["Titre", "Domaine_metier", "cluster_id", "cluster_nom"]].head(10))
