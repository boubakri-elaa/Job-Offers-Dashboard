import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer  # gardé si tu l'utilises plus tard


# ========================================
# CHEMINS
# ========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "offres_hellowork.csv")
INTERIM_DIR = os.path.join(BASE_DIR, "data", "interim")
os.makedirs(INTERIM_DIR, exist_ok=True)
CLEAN_PATH = os.path.join(INTERIM_DIR, "offres_hellowork_clean.csv")


print("📂 Chargement des données brutes...")
df = pd.read_csv(RAW_PATH, encoding="utf-8")
print(f"✅ {len(df)} offres chargées")


# ========================================
# NETTOYAGE DE BASE
# ========================================
print("\n🧹 Nettoyage des espaces et formatage...")

# Nettoyer les espaces pour toutes les colonnes texte
for col in ["Titre", "Entreprise", "Ville", "Contrat", "Date"]:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )


# ========================================
# TRAITEMENT DES DONNÉES MANQUANTES
# Important: On ne supprime PAS les lignes !
# ========================================
print("\n⚠️ Traitement des données manquantes (SANS suppression)...")

# 1) TITRE : Si manquant, mettre "Non spécifié"
df["Titre"] = (
    df["Titre"]
    .fillna("Non spécifié")
    .replace(["", "nan", "NaN"], "Non spécifié")
)

# 2) ENTREPRISE : Si manquant, mettre "Entreprise non communiquée"
df["Entreprise"] = (
    df["Entreprise"]
    .fillna("Entreprise non communiquée")
    .replace(["", "nan", "NaN"], "Entreprise non communiquée")
)

# 3) VILLE : Extraire ville et département (format: "Ville - 75")
ville_dep = df["Ville"].str.extract(r"^(?P<ville>.+?)\s*-\s*(?P<departement>\d+)$")
df["Ville_propre"] = ville_dep["ville"].fillna(df["Ville"]).fillna("Non spécifié").str.strip()
df["Departement"] = pd.to_numeric(ville_dep["departement"], errors="coerce")

# Si département manquant, mettre 0 (code pour "non spécifié")
df["Departement"] = df["Departement"].fillna(0).astype(int)

# 4) CONTRAT : Normaliser et gérer les manquants
df["Contrat_propre"] = (
    df["Contrat"]
    .fillna("NON_SPECIFIE")
    .astype(str)
    .str.upper()
    .str.replace(" ", "", regex=False)
    .replace(["", "NAN", "NONE"], "NON_SPECIFIE")
)

# 5) DATE : Garder telle quelle (on peut la traiter plus tard si besoin)
df["Date"] = (
    df["Date"]
    .fillna("Date inconnue")
    .replace(["", "nan", "NaN"], "Date inconnue")
)


# ========================================
# EXTRACTION DE MOTS-CLÉS DU TITRE
# Pour faciliter le clustering par domaine
# ========================================
print("\n🔍 Extraction des mots-clés métiers...")


def extraire_domaine(titre: str) -> str:
    """
    Détecte un domaine métier à partir du titre.
    Règles simples basées sur des mots-clés français.
    """
    t = str(titre).lower()

    # Restauration / Hôtellerie
    if any(m in t for m in ["cuisinier", "serveur", "restauration", "hôtel", "hotel", "chef de rang", "restaurant"]):
        return "Restauration"

    # Logistique / Transport
    if any(m in t for m in ["logistique", "chauffeur", "livreur", "transport", "pl de nuit", "magasinier", "cariste"]):
        return "Logistique"

    # BTP / Construction / Travaux
    if any(m in t for m in ["conducteur de travaux", "chantier", "géotechnique", "geotechnique", "ingénieur travaux", "travaux publics", "bâtiment", "batiment", "scierie"]):
        return "BTP"

    # Électricité / Énergie / Technique
    if any(m in t for m in ["électricien", "electricien", "électricité", "electricite", "électrique", "electric", "énergie", "energie", "technicien", "maintenance"]):
        return "Énergie / Technique"

    # Qualité / QHSE / Sécurité
    if any(m in t for m in ["qhse", "qse", "qualité", "qualite", "sécurité", "securite", "hse"]):
        return "Qualité / QHSE"

    # Finance / Assurance / Actuariat / Comptabilité
    if any(m in t for m in ["actuaire", "risques", "assurances", "assurance", "comptable", "comptabilité", "audit", "contrôle de gestion", "controle de gestion"]):
        return "Finance / Assurance"

    # Informatique / SI / Data / Digital
    if any(m in t for m in [
        "développeur", "developpeur", "développeuse", "developer",
        "informatique", "data", "si ", "système d'information", "systèmes d'information",
        "logiciel", "software", "it", "tech", "numérique", "digital", "progiciel"
    ]):
        return "Informatique"

    # Commerce / Vente / Magasin
    if any(m in t for m in [
        "commercial", "vente", "vendeur", "magasin", "magasinier",
        "conseiller de vente", "conseiller client", "relation client",
        "directeur de magasin", "responsable magasin"
    ]):
        return "Commerce"

    # Administration / Assistant / Support
    if any(m in t for m in [
        "assistant", "assistante", "administratif", "administrative",
        "gestionnaire", "secrétaire", "back office"
    ]):
        return "Administration"

    # Management / Direction / Chef de projet
    if any(m in t for m in [
        "manager", "responsable", "directeur", "directrice",
        "chef de projet", "chef de département", "chef d'équipe", "chef d equipe",
        "responsable agence", "responsable des projets"
    ]):
        return "Management"

    # Par défaut
    return "Autre"



# Appliquer la fonction
df["Domaine_metier"] = df["Titre"].apply(extraire_domaine)


# ========================================
# CRÉATION DU TEXTE COMPLET POUR ML
# ========================================
print("\n📝 Création du texte complet pour analyse ML...")

df["texte_complet"] = (
    df["Titre"].fillna("") + " " +
    df["Entreprise"].fillna("") + " " +
    df["Ville_propre"].fillna("") + " " +
    df["Contrat_propre"].fillna("") + " " +
    df["Domaine_metier"].fillna("")
).str.strip()


# ========================================
# CALCUL DE STATISTIQUES
# ========================================
print("\n📊 Statistiques des données nettoyées:")
print(f"  - Total offres: {len(df)}")
print(f"  - Villes uniques: {df['Ville_propre'].nunique()}")
print(f"  - Contrats uniques: {df['Contrat_propre'].nunique()}")
print(f"  - Domaines métiers:")
for domaine, count in df["Domaine_metier"].value_counts().items():
    print(f"      {domaine}: {count}")


# ========================================
# SAUVEGARDE
# ========================================
df.to_csv(CLEAN_PATH, index=False, encoding="utf-8")
print(f"\n✅ Données nettoyées sauvegardées dans: {CLEAN_PATH}")
print(f"✅ Forme finale: {df.shape}")
print("\n👀 Aperçu des 20 premières lignes:")
print(df[["Titre", "Entreprise", "Ville_propre", "Contrat_propre", "Domaine_metier"]].head(20))
