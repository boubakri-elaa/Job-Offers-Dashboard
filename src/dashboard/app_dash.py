import os
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go

# ========================================
# CHARGEMENT DES DONNÉES
# ========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
ML_PATH = os.path.join(BASE_DIR, "data", "processed", "offres_ml.csv")

df = pd.read_csv(ML_PATH, encoding="utf-8")

# ========================================
# PRÉPARATION DES OPTIONS DE FILTRES
# ========================================
villes_options = [{"label": v, "value": v} for v in sorted(df["Ville_propre"].dropna().unique())]
contrats_options = [{"label": c, "value": c} for c in sorted(df["Contrat_propre"].dropna().unique())]
domaines_options = [{"label": d, "value": d} for d in sorted(df["Domaine_metier"].unique())]
clusters_options = [{"label": f"Cluster {c}: {n}", "value": int(c)} 
                   for c, n in df.groupby("cluster_id")["cluster_nom"].first().items()]
salaire_options = [{"label": s, "value": s} for s in ["Bas", "Moyen", "Bon", "Élevé"]]

# ========================================
# COULEURS ET STYLE
# ========================================
BG_COLOR = "#f8fafc"
SIDEBAR_COLOR = "#ffffff"
CARD_COLOR = "#ffffff"
ACCENT = "#3b82f6"
ACCENT2 = "#8b5cf6"
ACCENT3 = "#10b981"
TEXT_COLOR = "#1e293b"
TEXT_LIGHT = "#64748b"

# ========================================
# APPLICATION DASH
# ========================================
app = Dash(__name__)

app.layout = html.Div(
    style={
        "fontFamily": "'Inter', 'Segoe UI', sans-serif",
        "backgroundColor": BG_COLOR,
        "display": "flex",
        "minHeight": "100vh",
    },
    children=[
        # ========================================
        # PANEL GAUCHE (SIDEBAR)
        # ========================================
        html.Div(
            style={
                "width": "320px",
                "backgroundColor": SIDEBAR_COLOR,
                "padding": "20px",
                "boxShadow": "2px 0 8px rgba(0,0,0,0.05)",
                "overflowY": "auto",
                "position": "fixed",
                "height": "100vh",
            },
            children=[
                # Titre
                html.H2(
                    "🎯 Filtres",
                    style={
                        "color": ACCENT,
                        "fontSize": "22px",
                        "marginBottom": "20px",
                        "fontWeight": "700",
                    }
                ),
                
                # Filtre: Ville
                html.Div(
                    style={"marginBottom": "20px"},
                    children=[
                        html.Label(
                            "📍 Ville",
                            style={"fontSize": "13px", "fontWeight": "600", "color": TEXT_COLOR, "marginBottom": "6px", "display": "block"}
                        ),
                        dcc.Dropdown(
                            id="filter-ville",
                            options=villes_options,
                            placeholder="Toutes les villes",
                            multi=True,
                            style={"fontSize": "13px"}
                        ),
                    ]
                ),
                
                # Filtre: Contrat
                html.Div(
                    style={"marginBottom": "20px"},
                    children=[
                        html.Label(
                            "📄 Type de contrat",
                            style={"fontSize": "13px", "fontWeight": "600", "color": TEXT_COLOR, "marginBottom": "6px", "display": "block"}
                        ),
                        dcc.Dropdown(
                            id="filter-contrat",
                            options=contrats_options,
                            placeholder="Tous les contrats",
                            multi=True,
                            style={"fontSize": "13px"}
                        ),
                    ]
                ),
                
                # Filtre: Domaine métier
                html.Div(
                    style={"marginBottom": "20px"},
                    children=[
                        html.Label(
                            "💼 Domaine métier",
                            style={"fontSize": "13px", "fontWeight": "600", "color": TEXT_COLOR, "marginBottom": "6px", "display": "block"}
                        ),
                        dcc.Dropdown(
                            id="filter-domaine",
                            options=domaines_options,
                            placeholder="Tous les domaines",
                            multi=True,
                            style={"fontSize": "13px"}
                        ),
                    ]
                ),
                
                # Filtre: Cluster
                html.Div(
                    style={"marginBottom": "20px"},
                    children=[
                        html.Label(
                            "🎨 Cluster métier",
                            style={"fontSize": "13px", "fontWeight": "600", "color": TEXT_COLOR, "marginBottom": "6px", "display": "block"}
                        ),
                        dcc.Dropdown(
                            id="filter-cluster",
                            options=clusters_options,
                            placeholder="Tous les clusters",
                            multi=True,
                            style={"fontSize": "13px"}
                        ),
                    ]
                ),
                
                # Filtre: Niveau de salaire
                html.Div(
                    style={"marginBottom": "20px"},
                    children=[
                        html.Label(
                            "💰 Niveau de salaire",
                            style={"fontSize": "13px", "fontWeight": "600", "color": TEXT_COLOR, "marginBottom": "6px", "display": "block"}
                        ),
                        dcc.Dropdown(
                            id="filter-salaire",
                            options=salaire_options,
                            placeholder="Tous les niveaux",
                            multi=True,
                            style={"fontSize": "13px"}
                        ),
                    ]
                ),
                
                # Checkbox: Métiers très demandés
                html.Div(
                    style={"marginBottom": "20px"},
                    children=[
                        html.Label(
                            "🔥 Options spéciales",
                            style={"fontSize": "13px", "fontWeight": "600", "color": TEXT_COLOR, "marginBottom": "8px", "display": "block"}
                        ),
                        dcc.Checklist(
                            id="filter-tres-demande",
                            options=[
                                {"label": " Métiers très demandés uniquement", "value": 1}
                            ],
                            value=[],
                            style={"fontSize": "12px", "color": TEXT_COLOR}
                        ),
                    ]
                ),
                
                # Bouton reset (optionnel)
                html.Button(
                    "🔄 Réinitialiser les filtres",
                    id="btn-reset",
                    n_clicks=0,
                    style={
                        "width": "100%",
                        "padding": "10px",
                        "backgroundColor": "#f1f5f9",
                        "border": "none",
                        "borderRadius": "8px",
                        "fontSize": "13px",
                        "fontWeight": "600",
                        "color": TEXT_COLOR,
                        "cursor": "pointer",
                        "marginTop": "10px",
                    }
                ),
            ]
        ),
        
        # ========================================
        # CONTENU PRINCIPAL (À DROITE)
        # ========================================
        html.Div(
            style={
                "marginLeft": "340px",
                "padding": "20px",
                "flex": 1,
            },
            children=[
                # Header
                html.Div(
                    style={"marginBottom": "20px"},
                    children=[
                        html.H1(
                            "📊 Dashboard BI - Offres d'Emploi Hellowork",
                            style={
                                "color": ACCENT,
                                "fontSize": "28px",
                                "fontWeight": "700",
                                "marginBottom": "5px",
                            }
                        ),
                        html.P(
                            "Analyse et visualisation dynamique des offres d'emploi avec Machine Learning",
                            style={"color": TEXT_LIGHT, "fontSize": "14px"}
                        ),
                    ]
                ),
                
                # Cartes statistiques
                html.Div(
                    id="stats-cards",
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(auto-fit, minmax(200px, 1fr))",
                        "gap": "15px",
                        "marginBottom": "20px",
                    }
                ),
                
                # Graphiques - Ligne 1
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "15px", "marginBottom": "15px"},
                    children=[
                        html.Div(
                            style={
                                "backgroundColor": CARD_COLOR,
                                "padding": "15px",
                                "borderRadius": "12px",
                                "boxShadow": "0 2px 8px rgba(0,0,0,0.06)",
                            },
                            children=[
                                html.H3("📈 Offres par domaine métier", style={"fontSize": "16px", "marginBottom": "10px", "color": TEXT_COLOR}),
                                dcc.Graph(id="graph-domaine", style={"height": "320px"}),
                            ]
                        ),
                        html.Div(
                            style={
                                "backgroundColor": CARD_COLOR,
                                "padding": "15px",
                                "borderRadius": "12px",
                                "boxShadow": "0 2px 8px rgba(0,0,0,0.06)",
                            },
                            children=[
                                html.H3("🎯 Distribution des clusters", style={"fontSize": "16px", "marginBottom": "10px", "color": TEXT_COLOR}),
                                dcc.Graph(id="graph-cluster-pie", style={"height": "320px"}),
                            ]
                        ),
                    ]
                ),
                
                # Graphiques - Ligne 2
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "15px", "marginBottom": "15px"},
                    children=[
                        html.Div(
                            style={
                                "backgroundColor": CARD_COLOR,
                                "padding": "15px",
                                "borderRadius": "12px",
                                "boxShadow": "0 2px 8px rgba(0,0,0,0.06)",
                            },
                            children=[
                                html.H3("💰 Distribution des niveaux de salaire", style={"fontSize": "16px", "marginBottom": "10px", "color": TEXT_COLOR}),
                                dcc.Graph(id="graph-salaire", style={"height": "320px"}),
                            ]
                        ),
                        html.Div(
                            style={
                                "backgroundColor": CARD_COLOR,
                                "padding": "15px",
                                "borderRadius": "12px",
                                "boxShadow": "0 2px 8px rgba(0,0,0,0.06)",
                            },
                            children=[
                                html.H3("🔥 Métiers très demandés vs autres", style={"fontSize": "16px", "marginBottom": "10px", "color": TEXT_COLOR}),
                                dcc.Graph(id="graph-demande", style={"height": "320px"}),
                            ]
                        ),
                    ]
                ),
                
                # Graphiques - Ligne 3
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1.5fr 1fr", "gap": "15px", "marginBottom": "15px"},
                    children=[
                        html.Div(
                            style={
                                "backgroundColor": CARD_COLOR,
                                "padding": "15px",
                                "borderRadius": "12px",
                                "boxShadow": "0 2px 8px rgba(0,0,0,0.06)",
                            },
                            children=[
                                html.H3("🏙️ Top 15 villes", style={"fontSize": "16px", "marginBottom": "10px", "color": TEXT_COLOR}),
                                dcc.Graph(id="graph-villes", style={"height": "380px"}),
                            ]
                        ),
                        html.Div(
                            style={
                                "backgroundColor": CARD_COLOR,
                                "padding": "15px",
                                "borderRadius": "12px",
                                "boxShadow": "0 2px 8px rgba(0,0,0,0.06)",
                            },
                            children=[
                                html.H3("📋 Types de contrat", style={"fontSize": "16px", "marginBottom": "10px", "color": TEXT_COLOR}),
                                dcc.Graph(id="graph-contrats", style={"height": "380px"}),
                            ]
                        ),
                    ]
                ),
                
                # Graphique - Ligne 4 (Popularité internationale)
                html.Div(
                    style={
                        "backgroundColor": CARD_COLOR,
                        "padding": "15px",
                        "borderRadius": "12px",
                        "boxShadow": "0 2px 8px rgba(0,0,0,0.06)",
                        "marginBottom": "15px",
                    },
                    children=[
                        html.H3("🌍 Top 20 métiers - Popularité internationale", style={"fontSize": "16px", "marginBottom": "10px", "color": TEXT_COLOR}),
                        dcc.Graph(id="graph-popularite", style={"height": "400px"}),
                    ]
                ),
            ]
        ),
    ]
)

# ========================================
# FONCTION DE FILTRAGE
# ========================================
def appliquer_filtres(df_in, villes, contrats, domaines, clusters, salaires, tres_demande):
    """Applique tous les filtres sélectionnés"""
    dff = df_in.copy()
    
    if villes:
        dff = dff[dff["Ville_propre"].isin(villes)]
    if contrats:
        dff = dff[dff["Contrat_propre"].isin(contrats)]
    if domaines:
        dff = dff[dff["Domaine_metier"].isin(domaines)]
    if clusters:
        dff = dff[dff["cluster_id"].isin(clusters)]
    if salaires:
        dff = dff[dff["niveau_salaire"].isin(salaires)]
    if tres_demande:
        dff = dff[dff["pred_tres_demande"] == 1]
    
    return dff

# ========================================
# CALLBACKS
# ========================================
@app.callback(
    Output("stats-cards", "children"),
    Output("graph-domaine", "figure"),
    Output("graph-cluster-pie", "figure"),
    Output("graph-salaire", "figure"),
    Output("graph-demande", "figure"),
    Output("graph-villes", "figure"),
    Output("graph-contrats", "figure"),
    Output("graph-popularite", "figure"),
    Input("filter-ville", "value"),
    Input("filter-contrat", "value"),
    Input("filter-domaine", "value"),
    Input("filter-cluster", "value"),
    Input("filter-salaire", "value"),
    Input("filter-tres-demande", "value"),
    Input("btn-reset", "n_clicks"),
)
def update_dashboard(villes, contrats, domaines, clusters, salaires, tres_demande, reset_clicks):
    # Appliquer les filtres
    tres_demande_bool = 1 in (tres_demande or [])
    dff = appliquer_filtres(df, villes, contrats, domaines, clusters, salaires, tres_demande_bool)
    
    # ========= CARTES STATS =========
    stats_cards = [
        html.Div(
            style={
                "backgroundColor": CARD_COLOR,
                "padding": "15px",
                "borderRadius": "10px",
                "boxShadow": "0 2px 6px rgba(0,0,0,0.05)",
            },
            children=[
                html.Div("📊 Total offres", style={"fontSize": "12px", "color": TEXT_LIGHT, "marginBottom": "5px"}),
                html.Div(f"{len(dff):,}".replace(",", " "), style={"fontSize": "24px", "fontWeight": "700", "color": ACCENT}),
            ]
        ),
        html.Div(
            style={
                "backgroundColor": CARD_COLOR,
                "padding": "15px",
                "borderRadius": "10px",
                "boxShadow": "0 2px 6px rgba(0,0,0,0.05)",
            },
            children=[
                html.Div("🏙️ Villes", style={"fontSize": "12px", "color": TEXT_LIGHT, "marginBottom": "5px"}),
                html.Div(f"{dff['Ville_propre'].nunique()}", style={"fontSize": "24px", "fontWeight": "700", "color": ACCENT2}),
            ]
        ),
        html.Div(
            style={
                "backgroundColor": CARD_COLOR,
                "padding": "15px",
                "borderRadius": "10px",
                "boxShadow": "0 2px 6px rgba(0,0,0,0.05)",
            },
            children=[
                html.Div("🔥 Très demandés", style={"fontSize": "12px", "color": TEXT_LIGHT, "marginBottom": "5px"}),
                html.Div(f"{(dff['pred_tres_demande']==1).mean()*100:.1f}%", style={"fontSize": "24px", "fontWeight": "700", "color": ACCENT3}),
            ]
        ),
        html.Div(
            style={
                "backgroundColor": CARD_COLOR,
                "padding": "15px",
                "borderRadius": "10px",
                "boxShadow": "0 2px 6px rgba(0,0,0,0.05)",
            },
            children=[
                html.Div("💰 Salaire élevé", style={"fontSize": "12px", "color": TEXT_LIGHT, "marginBottom": "5px"}),
                html.Div(f"{(dff['niveau_salaire']=='Élevé').mean()*100:.1f}%", style={"fontSize": "24px", "fontWeight": "700", "color": "#f59e0b"}),
            ]
        ),
    ]
    
    # ========= GRAPHIQUE 1: Domaines =========
    fig_domaine = px.bar(
        dff["Domaine_metier"].value_counts().reset_index(),
        x="count",
        y="Domaine_metier",
        orientation="h",
        color="count",
        color_continuous_scale="Blues",
        labels={"count": "Nombre d'offres", "Domaine_metier": ""},
    )
    fig_domaine.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    
    # ========= GRAPHIQUE 2: Clusters pie =========
    cluster_data = dff.groupby("cluster_nom").size().reset_index(name="count")
    fig_cluster_pie = px.pie(
        cluster_data,
        names="cluster_nom",
        values="count",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    fig_cluster_pie.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    
    # ========= GRAPHIQUE 3: Salaire =========
    salaire_order = ["Bas", "Moyen", "Bon", "Élevé"]
    salaire_counts = dff["niveau_salaire"].value_counts().reindex(salaire_order, fill_value=0).reset_index()
    salaire_counts.columns = ["Niveau", "count"]
    fig_salaire = px.bar(
        salaire_counts,
        x="Niveau",
        y="count",
        color="Niveau",
        color_discrete_map={"Bas": "#ef4444", "Moyen": "#f97316", "Bon": "#10b981", "Élevé": "#3b82f6"},
        labels={"count": "Nombre d'offres"},
    )
    fig_salaire.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    
    # ========= GRAPHIQUE 4: Demande =========
    demande_counts = dff["pred_tres_demande"].map({0: "Normal", 1: "Très demandé"}).value_counts().reset_index()
    demande_counts.columns = ["Type", "count"]
    fig_demande = px.bar(
        demande_counts,
        x="Type",
        y="count",
        color="Type",
        color_discrete_map={"Très demandé": ACCENT3, "Normal": "#94a3b8"},
        labels={"count": "Nombre d'offres"},
    )
    fig_demande.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    
    # ========= GRAPHIQUE 5: Top villes =========
    top_villes = dff["Ville_propre"].value_counts().head(15).reset_index()
    top_villes.columns = ["Ville", "count"]
    fig_villes = px.bar(
        top_villes,
        x="count",
        y="Ville",
        orientation="h",
        color="count",
        color_continuous_scale="Viridis",
        labels={"count": "Nombre d'offres", "Ville": ""},
    )
    fig_villes.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    
    # ========= GRAPHIQUE 6: Contrats =========
    contrat_data = dff["Contrat_propre"].value_counts().reset_index()
    contrat_data.columns = ["Contrat", "count"]
    fig_contrats = px.pie(
        contrat_data,
        names="Contrat",
        values="count",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig_contrats.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    
    # ========= GRAPHIQUE 7: Popularité internationale =========
    top_popularite = (
        dff.groupby("Titre")
        .agg({
            "score_popularite": "mean",
            "Titre": "count"
        })
        .rename(columns={"Titre": "nb_offres"})
        .sort_values("score_popularite", ascending=False)
        .head(20)
        .reset_index()
    )
    fig_popularite = px.bar(
        top_popularite,
        x="score_popularite",
        y="Titre",
        orientation="h",
        color="score_popularite",
        color_continuous_scale="RdYlGn",
        labels={"score_popularite": "Score de popularité", "Titre": ""},
        hover_data={"nb_offres": True},
    )
    fig_popularite.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    
    return stats_cards, fig_domaine, fig_cluster_pie, fig_salaire, fig_demande, fig_villes, fig_contrats, fig_popularite


if __name__ == "__main__":
    app.run(debug=True, port=8050)