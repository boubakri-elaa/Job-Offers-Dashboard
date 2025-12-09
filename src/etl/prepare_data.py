import os
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "offres_hellowork.csv")
INTERIM_DIR = os.path.join(BASE_DIR, "data", "interim")
os.makedirs(INTERIM_DIR, exist_ok=True)
CLEAN_PATH = os.path.join(INTERIM_DIR, "offres_hellowork_clean.csv")

# =========================
# 1) Load raw data
# =========================
df = pd.read_csv(RAW_PATH, encoding="utf-8")

# =========================
# 2) Basic cleaning with pandas
# =========================
text_cols = ["Titre", "Entreprise", "Ville", "Contrat", "Date"]
for col in text_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .replace({"": pd.NA})
    )

# remove rows without title or company
df = df.dropna(subset=["Titre", "Entreprise"])

# split location into city + departement
ville_dep = df["Ville"].str.extract(r"^(?P<ville>.+?)\s*-\s*(?P<departement>\d+)$")
df["Ville_propre"] = ville_dep["ville"].str.strip()
df["Departement"] = pd.to_numeric(ville_dep["departement"], errors="coerce")

# normalize contract type
df["Contrat_propre"] = df["Contrat"].str.upper().str.replace(" ", "")

# remove literal "nan" titles if any
df = df[df["Titre"].str.lower() != "nan"]

# combined text for ML later
df["texte_complet"] = (
    df["Titre"].fillna("") + " "
    + df["Entreprise"].fillna("") + " "
    + df["Ville_propre"].fillna("") + " "
    + df["Contrat_propre"].fillna("")
).str.strip()

# =========================
# 3) Preprocessing with scikit-learn
#    (imputation + encoding)
# =========================
cat_features = ["Contrat_propre"]
num_features = ["Departement"]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features),
    ],
    remainder="drop",
)

# fit + transform to obtain a numerical feature matrix for later ML
X_prepared = preprocessor.fit_transform(df)

# (optionally, you could save X_prepared to disk later for models)

# =========================
# 4) Save cleaned pandas dataset
# =========================
df.to_csv(CLEAN_PATH, index=False, encoding="utf-8")
print("✅ Données nettoyées enregistrées dans :", CLEAN_PATH)
print(df.head())
print(df.shape)
