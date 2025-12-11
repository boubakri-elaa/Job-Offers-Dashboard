import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ========================================
# CHEMINS
# ========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CLUSTERS_PATH = os.path.join(BASE_DIR, "data", "processed", "offres_clusters.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)
ML_PATH = os.path.join(PROCESSED_DIR, "offres_ml.csv")


# ========================================
# CHARGEMENT DES DONNÉES
# ========================================
print("📂 Chargement des données avec clusters...")
df = pd.read_csv(CLUSTERS_PATH, encoding="utf-8")
print(f"✅ {len(df)} offres chargées")


# ========================================
# PRÉTRAITEMENT TEXTE (même logique que clustering)
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

df["texte_ml"] = df["texte_complet"].fillna("").apply(nettoyer_texte)


# ========================================
# 1) PRÉDICTION : MÉTIER TRÈS DEMANDÉ
# ========================================
print("\n🎯 Prédiction: Métiers très demandés...")

# Comptage par titre
titre_counts = df["Titre"].value_counts()

# Top 20% des métiers les plus fréquents
seuil = titre_counts.quantile(0.80)

df["metier_tres_demande"] = df["Titre"].map(
    lambda x: 1 if titre_counts.get(x, 0) >= seuil else 0
)

print(f"   Seuil pour 'très demandé': {seuil:.0f} offres")
print(f"   Métiers très demandés: {df['metier_tres_demande'].sum()} offres")
print(f"   Métiers normaux: {(df['metier_tres_demande'] == 0).sum()} offres")

X = df["texte_ml"]
y = df["metier_tres_demande"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n🤖 Entraînement du modèle de classification...")

vectorizer_clf = TfidfVectorizer(max_features=1000, min_df=2)
X_train_vec = vectorizer_clf.fit_transform(X_train)
X_test_vec = vectorizer_clf.transform(X_test)

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_vec, y_train)

y_pred = clf.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Précision (accuracy): {acc:.3f}")

print("\n🧾 Classification report:")
print(classification_report(y_test, y_pred))

# Prédiction sur toutes les offres
X_all_vec = vectorizer_clf.transform(X)
df["pred_tres_demande"] = clf.predict(X_all_vec)


# ========================================
# 2) ESTIMATION DU NIVEAU DE SALAIRE (heuristique)
# ========================================
print("\n💰 Estimation du niveau de salaire...")

def estimer_salaire(row):
    score = 50  # base 0-100

    contrat = str(row["Contrat_propre"]).upper()
    if "CDI" in contrat:
        score += 20
    elif "CDD" in contrat:
        score += 10

    domaine = str(row["Domaine_metier"])
    if domaine == "Informatique":
        score += 25
    elif domaine == "Santé":
        score += 15
    elif domaine == "Industrie":
        score += 10
    elif domaine == "Commerce":
        score += 5

    titre = str(row["Titre"]).lower()
    if any(m in titre for m in ["manager", "directeur", "responsable", "chef", "lead"]):
        score += 20
    if any(m in titre for m in ["senior", "expert", "ingénieur", "developpeur", "développeur"]):
        score += 15
    if any(m in titre for m in ["junior", "assistant", "stagiaire"]):
        score -= 10

    return max(0, min(100, score))

df["score_salaire"] = df.apply(estimer_salaire, axis=1)

df["niveau_salaire"] = pd.cut(
    df["score_salaire"],
    bins=[0, 40, 60, 80, 100],
    labels=["Bas", "Moyen", "Bon", "Élevé"]
)

print("   Distribution des niveaux de salaire:")
for niveau, count in df["niveau_salaire"].value_counts().sort_index().items():
    print(f"      {niveau}: {count} offres")


# ========================================
# 3) SCORE DE POPULARITÉ
# ========================================
print("\n🌍 Calcul du score de popularité...")

df["score_popularite"] = (
    df.groupby("Titre")["Titre"].transform("count") / len(df) * 100 +
    df["pred_tres_demande"] * 30 +
    df["Domaine_metier"].map({
        "Informatique": 30,
        "Santé": 25,
        "Commerce": 15,
        "Logistique": 10,
        "Administration": 10,
        "Industrie": 15,
        "Restauration": 5,
        "Autre": 5,
        "BTP": 15,
        "Énergie / Technique": 20,
        "Finance / Assurance": 20,
        "Management": 15,
        "Qualité / QHSE": 10,
    }).fillna(5)
).round(1)

df["score_popularite"] = (
    (df["score_popularite"] - df["score_popularite"].min()) /
    (df["score_popularite"].max() - df["score_popularite"].min()) * 100
).round(1)


# ========================================
# AFFICHAGE DES RÉSULTATS
# ========================================
print("\n📊 Top 10 métiers les plus fréquents:")
top_metiers = (
    df.groupby("Titre")
    .agg({
        "Titre": "count",
        "pred_tres_demande": "mean",
        "score_popularite": "mean",
        "niveau_salaire": lambda x: x.mode()[0] if len(x) > 0 else "Moyen"
    })
    .rename(columns={"Titre": "Nombre_offres"})
    .sort_values("Nombre_offres", ascending=False)
    .head(10)
)
print(top_metiers)

# ========================================
# SAUVEGARDE
# ========================================
df.to_csv(ML_PATH, index=False, encoding="utf-8")
print(f"\n✅ Fichier enrichi (ML complet) sauvegardé dans: {ML_PATH}")
print("✅ Nouvelles colonnes ajoutées :")
print("   - metier_tres_demande (0/1, vrai label)")
print("   - pred_tres_demande (0/1, prédiction)")
print("   - score_salaire (0-100)")
print("   - niveau_salaire (Bas/Moyen/Bon/Élevé)")
print("   - score_popularite (0-100)")

print("\n👀 Aperçu des données finales:")
print(df[["Titre", "Domaine_metier", "pred_tres_demande", "niveau_salaire", "score_popularite"]].head(10))
