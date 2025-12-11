# 📊 Projet BI - Analyse des Offres d'Emploi Hellowork

Ce projet analyse automatiquement les offres d'emploi publiées sur Hellowork et fournit un **dashboard interactif** pour explorer les données.

---

## 🎯 Objectif
- Collecter et nettoyer les offres d'emploi  
- Identifier les domaines métiers et regrouper les offres similaires  
- Estimer les métiers les plus demandés et le niveau de salaire  
- Visualiser les données dans un dashboard interactif  

---

## 🗂️ Structure du projet
Projet_Bi/
├── data/
│ ├── raw/ # Données brutes du scraping
│ │ └── offres_hellowork.csv
│ ├── interim/ # Données nettoyées
│ │ └── offres_hellowork_clean.csv
│ └── processed/ # Données enrichies (clusters + ML)
│ ├── offres_clusters.csv
│ └── offres_ml.csv
├── src/
│ ├── scraping/
│ │ └── scrape_hellowork.py
│ ├── etl/
│ │ └── prepare_data.py
│ ├── ml/
│ │ ├── clustering.py
│ │ └── classification.py
│ └── dashboard/
│ └── app_dash.py
└── README.md

---

## ▶️ Comment utiliser
1. Lancer le script pour récupérer les offres  
2. Nettoyer et préparer les données  
3. Effectuer le regroupement par métiers et les prédictions  
4. Ouvrir le dashboard pour explorer les résultats  

Le dashboard sera accessible à l'adresse : [http://127.0.0.1:8050](http://127.0.0.1:8050)

---

## 🎨 Dashboard
- Filtres dynamiques : ville, type de contrat, domaine métier, cluster, niveau de salaire, métiers populaires  
- Visualisations : graphiques et cartes pour explorer les tendances  

---

## 👥 Auteur
Projet BI - Analyse des Offres d'Emploi  
Cours de Business Intelligence  

Bon visionnage ! 🚀


