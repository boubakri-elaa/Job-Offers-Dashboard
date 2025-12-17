import os
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go

# ========================================
# CONFIGURATION & CHARGEMENT
# ========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
ML_PATH = os.path.join(BASE_DIR, "data", "processed", "offres_ml.csv")

df = pd.read_csv(ML_PATH, encoding="utf-8")

# Colonnes : Titre, Entreprise, Ville, Contrat, Date, Pays,
# cluster_id, cluster_category, cluster_keywords,
# confidence_score, experience_level, experience_confidence,
# salary_level, salary_score, salary_confidence [file:3]

# Harmonisation
df["Ville_propre"] = df["Ville"]
df["Contrat_propre"] = df["Contrat"]
df["Domaine_metier"] = df["cluster_category"]
df["niveau_salaire"] = df["salary_level"]

df["pred_tres_demande"] = (df["confidence_score"] > 0.6).astype(int)
df["score_popularite"] = df["salary_score"]

HAS_PAYS = "Pays" in df.columns

# Nettoyage pour l’axe temps (nombre de jours depuis la publication)
def extraire_jours(s):
    if isinstance(s, str):
        s = s.strip().lower()
        if "jour" in s:
            try:
                return float(s.split("il y a")[1].split("jour")[0].strip())
            except Exception:
                return None
        if "heure" in s:
            try:
                h = float(s.split("il y a")[1].split("heure")[0].strip())
                return h / 24.0
            except Exception:
                return None
    return None

df["jours_depuis"] = df["Date"].apply(extraire_jours)
df["jours_depuis"] = pd.to_numeric(df["jours_depuis"], errors="coerce")
max_jours = df["jours_depuis"].max() if pd.notnull(df["jours_depuis"].max()) else 0.0
df["jours_depuis"] = df["jours_depuis"].fillna(max_jours)

# ========================================
# PALETTE & STYLES
# ========================================
COLORS = {
    'primary': '#1E3A8A',
    'secondary': '#2563EB',
    'accent': '#7C3AED',
    'success': '#10B981',
    'warning': '#F59E0B',
    'danger': '#EF4444',
    'info': '#06B6D4',
    'light_blue': '#EEF2FF',
    'light_green': '#ECFDF3',
    'light_orange': '#FFFBEB',
    'gray_50': '#F9FAFB',
    'gray_100': '#F3F4F6',
    'gray_200': '#E5E7EB',
    'gray_400': '#9CA3AF',
    'gray_600': '#4B5563',
    'gray_900': '#111827',
    'gray_700': "#000473",   # ← ajoute cette ligne

}

CATEGORICAL = ['#2563EB', '#7C3AED', '#10B981', '#F59E0B', '#EF4444',
               '#06B6D4', '#EC4899', '#6366F1', '#14B8A6']

CARD_STYLE = {
    'backgroundColor': 'white',
    'borderRadius': '10px',
    'boxShadow': '0 4px 10px rgba(15,23,42,0.08)',  # soft 3D
    'border': f"1px solid {COLORS['gray_200']}",
    'padding': '18px',
}

SIDEBAR_WRAPPER = {
    'position': 'fixed',
    'top': '0',
    'left': '0',
    'bottom': '0',
    'width': '320px',
    'padding': '18px',
    'backgroundColor': COLORS['gray_50'],
    'borderRight': f"1px solid {COLORS['gray_200']}",
    'overflowY': 'auto',
}

SIDEBAR_CARD = {
    'backgroundColor': 'white',
    'borderRadius': '10px',
    'boxShadow': '0 4px 10px rgba(15,23,42,0.06)',
    'border': f"1px solid {COLORS['gray_200']}",
    'padding': '16px',
}

# ========================================
# OPTIONS FILTRES
# ========================================
pays_options = [{"label": p, "value": p} for p in sorted(df["Pays"].dropna().unique())] if HAS_PAYS else []
villes_options = [{"label": v, "value": v} for v in sorted(df["Ville_propre"].dropna().unique())]
contrats_options = [{"label": c, "value": c} for c in sorted(df["Contrat_propre"].dropna().unique())]
domaines_options = [{"label": d, "value": d} for d in sorted(df["Domaine_metier"].dropna().unique())]
salaire_options = [{"label": s, "value": s} for s in sorted(df["niveau_salaire"].dropna().unique())]

# ========================================
# APP
# ========================================
app = Dash(__name__)
server = app.server   # juste après la création de app

def create_filter_section(title, filter_id, options, placeholder, multi=True):
    return html.Div([
        html.Label(
            title,
            style={
                'fontSize': '12px',
                'fontWeight': '600',
                'color': COLORS['gray_700'],
                'marginBottom': '6px',
                'display': 'block',
            }
        ),
        dcc.Dropdown(
            id=filter_id,
            options=options,
            placeholder=placeholder,
            multi=multi,
            style={
                'marginBottom': '12px',
                'fontSize': '13px',
            },
        ),
    ])

sidebar = html.Div(
    style=SIDEBAR_WRAPPER,
    children=[
        html.Div(
            style=SIDEBAR_CARD,
            children=[
                html.H2(
                    "BI Jobs Dashboard",
                    style={
                        'color': COLORS['primary'],
                        'fontSize': '18px',
                        'fontWeight': '700',
                        'marginBottom': '4px',
                    }
                ),
                html.P(
                    "Vue globale des offres Hellowork.",
                    style={
                        'color': COLORS['gray_600'],
                        'fontSize': '12px',
                        'marginBottom': '14px',
                    }
                ),
                html.H3(
                    "Géographie",
                    style={'fontSize': '13px', 'fontWeight': '600',
                           'color': COLORS['gray_900'], 'marginBottom': '6px'}
                ),
                create_filter_section("Pays", "filter-pays", pays_options, "Tous les pays") if HAS_PAYS else html.Div(),
                create_filter_section("Ville", "filter-ville", villes_options, "Toutes les villes"),

                html.H3(
                    "Profil d’emploi",
                    style={'fontSize': '13px', 'fontWeight': '600',
                           'color': COLORS['gray_900'], 'marginBottom': '6px', 'marginTop': '4px'}
                ),
                create_filter_section("Type de contrat", "filter-contrat", contrats_options, "Tous les contrats"),
                create_filter_section("Domaine métier", "filter-domaine", domaines_options, "Tous les domaines"),
                create_filter_section("Niveau de salaire", "filter-salaire", salaire_options, "Tous les niveaux"),

                html.H3(
                    "Options",
                    style={'fontSize': '13px', 'fontWeight': '600',
                           'color': COLORS['gray_900'], 'marginBottom': '6px', 'marginTop': '4px'}
                ),
                dcc.Checklist(
                    id="filter-tres-demande",
                    options=[{"label": "Métiers très demandés uniquement", "value": 1}],
                    value=[],
                    style={'fontSize': '12px', 'color': COLORS['gray_700']},
                    inputStyle={'marginRight': '6px'},
                ),

                html.Button(
                    "Réinitialiser les filtres",
                    id="btn-reset",
                    n_clicks=0,
                    style={
                        'width': '100%',
                        'padding': '9px 12px',
                        'backgroundColor': COLORS['primary'],
                        'color': 'white',
                        'border': 'none',
                        'borderRadius': '999px',
                        'fontSize': '13px',
                        'fontWeight': '600',
                        'cursor': 'pointer',
                        'marginTop': '10px',
                        'boxShadow': '0 4px 8px rgba(30,64,175,0.35)',
                    }
                ),
            ]
        )
    ]
)

app.layout = html.Div([
    sidebar,
    html.Div(
        style={
            'marginLeft': '352px',
            'padding': '20px',
            'backgroundColor': COLORS['gray_50'],
            'minHeight': '100vh',
        },
        children=[
            # HEADER
            html.Div(
                style={**CARD_STYLE, 'marginBottom': '20px'},
                children=[
                    html.Div(
                        "Tableau de bord – Offres d’emploi",
                        style={
                            'fontSize': '12px',
                            'letterSpacing': '0.08em',
                            'textTransform': 'uppercase',
                            'color': COLORS['gray_600'],
                            'marginBottom': '4px',
                        }
                    ),
                    html.H1(
                        "Analyse intelligente du marché Hellowork",
                        style={
                            'color': COLORS['primary'],
                            'fontSize': '24px',
                            'fontWeight': '700',
                            'marginBottom': '4px',
                        }
                    ),
                    html.P(
                        "Visualisation des offres par pays, domaines, salaires, expérience et types de contrat.",
                        style={'color': COLORS['gray_600'], 'fontSize': '13px', 'margin': '0'}
                    ),
                ]
            ),

            # KPI CARDS
            html.Div(
                id="kpi-cards",
                style={
                    'display': 'grid',
                    'gridTemplateColumns': 'repeat(auto-fit, minmax(200px, 1fr))',
                    'gap': '14px',
                    'marginBottom': '18px',
                }
            ),

            # LIGNES GRAPHIQUES (même structure que ton code précédent)
            html.Div(
                style={
                    'display': 'grid',
                    'gridTemplateColumns': '2fr 1fr',
                    'gap': '16px',
                    'marginBottom': '16px',
                },
                children=[
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.H3(
                                "Top 10 domaines de métiers",
                                style={'fontSize': '14px', 'fontWeight': '600',
                                       'marginBottom': '6px', 'color': COLORS['gray_900']}
                            ),
                            html.P(
                                "Part relative des principaux domaines dans le volume total d’offres.",
                                style={'fontSize': '12px', 'color': COLORS['gray_600'],
                                       'marginBottom': '10px'}
                            ),
                            dcc.Graph(id="graph-top-domaines", style={'height': '360px'}),
                        ]
                    ),
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.H3(
                                "Répartition des niveaux de salaire",
                                style={'fontSize': '14px', 'fontWeight': '600',
                                       'marginBottom': '6px', 'color': COLORS['gray_900']}
                            ),
                            dcc.Graph(id="graph-salaire-funnel", style={'height': '360px'}),
                        ]
                    ),
                ]
            ),

            html.Div(
                style={
                    'display': 'grid',
                    'gridTemplateColumns': '2fr 1fr',
                    'gap': '16px',
                    'marginBottom': '16px',
                },
                children=[
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.H3(
                                "Offres par pays",
                                style={'fontSize': '14px', 'fontWeight': '600',
                                       'marginBottom': '6px', 'color': COLORS['gray_900']}
                            ),
                            dcc.Graph(id="graph-offres-pays", style={'height': '340px'}),
                        ]
                    ),
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.H3(
                                "Répartition des catégories de métiers",
                                style={'fontSize': '14px', 'fontWeight': '600',
                                       'marginBottom': '6px', 'color': COLORS['gray_900']}
                            ),
                            dcc.Graph(id="graph-categories-pie", style={'height': '340px'}),
                        ]
                    ),
                ]
            ),

            html.Div(
                style={
                    'display': 'grid',
                    'gridTemplateColumns': '1fr 1fr',
                    'gap': '16px',
                    'marginBottom': '16px',
                },
                children=[
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.H3(
                                "Salaire moyen par pays",
                                style={'fontSize': '14px', 'fontWeight': '600',
                                       'marginBottom': '6px', 'color': COLORS['gray_900']}
                            ),
                            dcc.Graph(id="graph-salaire-pays", style={'height': '320px'}),
                        ]
                    ),
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.H3(
                                "Offres publiées dans le temps",
                                style={'fontSize': '14px', 'fontWeight': '600',
                                       'marginBottom': '6px', 'color': COLORS['gray_900']}
                            ),
                            dcc.Graph(id="graph-offres-temps", style={'height': '320px'}),
                        ]
                    ),
                ]
            ),

            html.Div(
                style={
                    'display': 'grid',
                    'gridTemplateColumns': '1fr 1fr',
                    'gap': '16px',
                    'marginBottom': '16px',
                },
                children=[
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.H3(
                                "Répartition des niveaux d’expérience",
                                style={'fontSize': '14px', 'fontWeight': '600',
                                       'marginBottom': '6px', 'color': COLORS['gray_900']}
                            ),
                            dcc.Graph(id="graph-experience-donut", style={'height': '320px'}),
                        ]
                    ),
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.H3(
                                "Salaire selon l’expérience",
                                style={'fontSize': '14px', 'fontWeight': '600',
                                       'marginBottom': '6px', 'color': COLORS['gray_900']}
                            ),
                            dcc.Graph(id="graph-salaire-experience", style={'height': '320px'}),
                        ]
                    ),
                ]
            ),

            html.Div(
                style={
                    'display': 'grid',
                    'gridTemplateColumns': '1fr 1fr',
                    'gap': '16px',
                    'marginBottom': '16px',
                },
                children=[
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.H3(
                                "Top 10 villes en nombre d’offres",
                                style={'fontSize': '14px', 'fontWeight': '600',
                                       'marginBottom': '6px', 'color': COLORS['gray_900']}
                            ),
                            dcc.Graph(id="graph-villes", style={'height': '320px'}),
                        ]
                    ),
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.H3(
                                "Répartition des types de contrat",
                                style={'fontSize': '14px', 'fontWeight': '600',
                                       'marginBottom': '6px', 'color': COLORS['gray_900']}
                            ),
                            dcc.Graph(id="graph-contrats-pie", style={'height': '320px'}),
                        ]
                    ),
                ]
            ),

            html.Div(
                style={
                    'display': 'grid',
                    'gridTemplateColumns': '1.4fr 0.8fr',
                    'gap': '16px',
                },
                children=[
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.H3(
                                "Niveaux de salaire par pays",
                                style={'fontSize': '14px', 'fontWeight': '600',
                                       'marginBottom': '6px', 'color': COLORS['gray_900']}
                            ),
                            dcc.Graph(id="graph-salaire-pays-stacked", style={'height': '320px'}),
                        ]
                    ),
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.H3(
                                "Part des métiers très demandés",
                                style={'fontSize': '14px', 'fontWeight': '600',
                                       'marginBottom': '6px', 'color': COLORS['gray_900']}
                            ),
                            dcc.Graph(id="graph-gauge-demande", style={'height': '320px'}),
                        ]
                    ),
                ]
            ),
        ]
    )
])

# ========================================
# CALLBACK
# ========================================
@app.callback(
    [
        Output("kpi-cards", "children"),
        Output("graph-top-domaines", "figure"),
        Output("graph-salaire-funnel", "figure"),
        Output("graph-offres-pays", "figure"),
        Output("graph-categories-pie", "figure"),
        Output("graph-salaire-pays", "figure"),
        Output("graph-offres-temps", "figure"),
        Output("graph-experience-donut", "figure"),
        Output("graph-salaire-experience", "figure"),
        Output("graph-villes", "figure"),
        Output("graph-contrats-pie", "figure"),
        Output("graph-salaire-pays-stacked", "figure"),
        Output("graph-gauge-demande", "figure"),
    ],
    [
        Input("filter-ville", "value"),
        Input("filter-contrat", "value"),
        Input("filter-domaine", "value"),
        Input("filter-pays", "value") if HAS_PAYS else Input("filter-contrat", "value"),
        Input("filter-salaire", "value"),
        Input("filter-tres-demande", "value"),
        Input("btn-reset", "n_clicks"),
    ]
)
def update_dashboard(villes, contrats, domaines, pays_or_dummy, salaires, tres_demande, n_reset):
    pays = pays_or_dummy if HAS_PAYS else None

    # Reset : on ignore les filtres quand on clique
    if n_reset:
        villes = None
        contrats = None
        domaines = None
        salaires = None
        tres_demande = []
        if HAS_PAYS:
            pays = None

    dff = df.copy()
    if villes:
        dff = dff[dff["Ville_propre"].isin(villes)]
    if contrats:
        dff = dff[dff["Contrat_propre"].isin(contrats)]
    if domaines:
        dff = dff[dff["Domaine_metier"].isin(domaines)]
    if HAS_PAYS and pays:
        dff = dff[dff["Pays"].isin(pays)]
    if salaires:
        dff = dff[dff["niveau_salaire"].isin(salaires)]
    if tres_demande and 1 in tres_demande:
        dff = dff[dff["pred_tres_demande"] == 1]

    # KPI
    total_offres = len(dff)
    taux_demande = (dff["pred_tres_demande"] == 1).mean() * 100 if total_offres > 0 else 0.0

    kpi_data = [
        ("Total d’offres", f"{total_offres:,}".replace(",", " "), COLORS['light_blue'], COLORS['primary']),
        ("Pays couverts", f"{dff['Pays'].nunique() if HAS_PAYS else 1}", COLORS['light_green'], COLORS['success']),
        ("Domaines distincts", f"{dff['Domaine_metier'].nunique()}", COLORS['light_blue'], COLORS['accent']),
        ("Offres très demandées", f"{taux_demande:.1f}%", COLORS['light_orange'], COLORS['warning']),
    ]

    kpi_cards = []
    for label, value, bg, accent in kpi_data:
        kpi_cards.append(
            html.Div(
                style={
                    'backgroundColor': bg,
                    'padding': '14px 16px',
                    'borderRadius': '10px',
                    'border': f"1px solid {COLORS['gray_200']}",
                    'boxShadow': '0 3px 8px rgba(15,23,42,0.06)',
                },
                children=[
                    html.Div(label, style={
                        'fontSize': '12px',
                        'color': COLORS['gray_600'],
                        'marginBottom': '4px',
                        'fontWeight': '600'
                    }),
                    html.Div(value, style={
                        'fontSize': '20px',
                        'fontWeight': '700',
                        'color': accent
                    }),
                ]
            )
        )

    # 1) Top 10 domaines (en %)
    series_dom = dff["Domaine_metier"].fillna("")
    masque_autre = ~series_dom.str.lower().str.contains("autre", na=False)
    domaines_filtre = series_dom[masque_autre]

    domaines_count = domaines_filtre.value_counts()
    total_dom = domaines_count.sum()
    if total_dom > 0:
        domaines_pct = (domaines_count / total_dom * 100).head(10)
    else:
        domaines_pct = pd.Series([0.0], index=["Aucun domaine"])

    fig_top_domaines = go.Figure(
        data=[go.Bar(
            x=domaines_pct.index,
            y=domaines_pct.values,
            marker_color=COLORS['accent'],
            text=[f"{v:.1f} %" for v in domaines_pct.values],
            textposition="outside",
            textfont=dict(color=COLORS['gray_900'], size=11),
        )]
    )
    fig_top_domaines.update_layout(
        xaxis_title="Domaine de métier",
        yaxis_title="Part des offres (%)",
        margin=dict(l=40, r=20, t=20, b=110),
        xaxis_tickangle=-45,
        plot_bgcolor='white',
        paper_bgcolor='white',
    )

    # 2) Niveaux de salaires (funnel)
    ordre_salaires = ["Très_Faible", "Faible", "Moyen", "Élevé", "Très_Élevé"]
    sal_counts = dff["niveau_salaire"].value_counts().reindex(ordre_salaires, fill_value=0)
    fig_salaire_funnel = go.Figure(go.Funnel(
        y=ordre_salaires,
        x=sal_counts.values,
        textinfo="value+percent initial",
        marker=dict(color=['#F97373', '#FDBA74', '#60A5FA', '#4F46E5', '#22C55E'])
    ))
    fig_salaire_funnel.update_layout(
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor='white',
        plot_bgcolor='white',
    )

    # 3) Offres par pays
    if HAS_PAYS:
        pays_count = dff["Pays"].value_counts()
        fig_offres_pays = go.Figure(go.Bar(
            x=pays_count.index,
            y=pays_count.values,
            marker_color=COLORS['secondary']
        ))
        fig_offres_pays.update_layout(
            xaxis_title="Pays",
            yaxis_title="Nombre d’offres",
            margin=dict(l=40, r=20, t=10, b=80),
            xaxis_tickangle=-45,
            plot_bgcolor='white',
            paper_bgcolor='white',
        )
    else:
        fig_offres_pays = go.Figure()

    # 4) Répartition des catégories (pie)
    cat_counts = domaines_filtre.value_counts()
    if cat_counts.empty:
        cat_counts = pd.Series([1], index=["Aucun domaine"])
    fig_categories_pie = go.Figure(go.Pie(
        labels=cat_counts.index,
        values=cat_counts.values,
        hole=0.4,
        marker=dict(colors=CATEGORICAL)
    ))
    fig_categories_pie.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        paper_bgcolor='white',
        plot_bgcolor='white',
    )

    # 5) Salaire moyen par pays
    if HAS_PAYS:
        salaire_pays = dff.groupby("Pays")["salary_score"].mean().sort_values()
        fig_salaire_pays = go.Figure(go.Bar(
            x=salaire_pays.values,
            y=salaire_pays.index,
            orientation='h',
            marker_color=COLORS['primary']
        ))
        fig_salaire_pays.update_layout(
            xaxis_title="Salaire moyen (score)",
            yaxis_title="Pays",
            margin=dict(l=80, r=20, t=10, b=40),
            paper_bgcolor='white',
            plot_bgcolor='white',
        )
    else:
        fig_salaire_pays = go.Figure()

    # 6) Offres dans le temps
    temps_data = dff.groupby("jours_depuis")["Titre"].count().reset_index()
    temps_data = temps_data.sort_values("jours_depuis", ascending=True)
    temps_data["cumul"] = temps_data["Titre"].cumsum()
    fig_offres_temps = go.Figure(go.Scatter(
        x=temps_data["jours_depuis"],
        y=temps_data["cumul"],
        mode="lines+markers",
        line=dict(color=COLORS['secondary'], width=2),
        marker=dict(size=5),
    ))
    fig_offres_temps.update_layout(
        xaxis_title="Nombre de jours depuis la publication (faible = récent)",
        yaxis_title="Offres cumulées",
        margin=dict(l=40, r=20, t=10, b=40),
        paper_bgcolor='white',
        plot_bgcolor='white',
    )

    # 7) Expérience (donut)
    exp_counts = dff["experience_level"].value_counts()
    fig_experience_donut = go.Figure(go.Pie(
        labels=exp_counts.index,
        values=exp_counts.values,
        hole=0.45,
        marker=dict(colors=CATEGORICAL)
    ))
    fig_experience_donut.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        paper_bgcolor='white',
        plot_bgcolor='white',
    )

    # 8) Salaire vs expérience (box)
    fig_salaire_experience = go.Figure()
    for lvl, col in zip(exp_counts.index, CATEGORICAL):
        fig_salaire_experience.add_trace(go.Box(
            x=[lvl] * len(dff[dff["experience_level"] == lvl]),
            y=dff.loc[dff["experience_level"] == lvl, "salary_score"],
            name=lvl,
            marker_color=col,
            boxmean='sd',
        ))
    fig_salaire_experience.update_layout(
        xaxis_title="Niveau d’expérience",
        yaxis_title="Score de salaire",
        margin=dict(l=40, r=20, t=10, b=40),
        showlegend=False,
        paper_bgcolor='white',
        plot_bgcolor='white',
    )

    # 9) Top 10 villes
    villes_count = dff["Ville_propre"].value_counts().head(10)
    fig_villes = go.Figure(go.Bar(
        x=villes_count.index,
        y=villes_count.values,
        marker_color=COLORS['accent']
    ))
    fig_villes.update_layout(
        xaxis_title="Ville",
        yaxis_title="Nombre d’offres",
        margin=dict(l=40, r=20, t=10, b=100),
        xaxis_tickangle=-45,
        paper_bgcolor='white',
        plot_bgcolor='white',
    )

    # 10) Types de contrat
    contrats_count = dff["Contrat_propre"].value_counts()
    fig_contrats_pie = go.Figure(go.Pie(
        labels=contrats_count.index,
        values=contrats_count.values,
        hole=0.4,
        marker=dict(colors=CATEGORICAL)
    ))
    fig_contrats_pie.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        paper_bgcolor='white',
        plot_bgcolor='white',
    )

    # 11) Niveaux de salaire par pays (stacked)
    if HAS_PAYS:
        salaire_pays_stack = dff.groupby(["Pays", "niveau_salaire"])["Titre"].count().reset_index()
        salaire_pays_stack = salaire_pays_stack.pivot(index="Pays", columns="niveau_salaire",
                                                      values="Titre").fillna(0)
        salaire_pays_stack = salaire_pays_stack.reindex(columns=ordre_salaires).fillna(0)

        fig_salaire_pays_stacked = go.Figure()
        for i, lvl in enumerate(salaire_pays_stack.columns):
            fig_salaire_pays_stacked.add_trace(go.Bar(
                x=salaire_pays_stack.index,
                y=salaire_pays_stack[lvl],
                name=lvl,
                marker_color=CATEGORICAL[i % len(CATEGORICAL)]
            ))
        fig_salaire_pays_stacked.update_layout(
            barmode='stack',
            xaxis_title="Pays",
            yaxis_title="Nombre d’offres",
            margin=dict(l=40, r=20, t=10, b=80),
            xaxis_tickangle=-45,
            paper_bgcolor='white',
            plot_bgcolor='white',
        )
    else:
        fig_salaire_pays_stacked = go.Figure()

    # 12) Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=taux_demande,
        title={'text': "Offres très demandées (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': COLORS['primary']},
            'steps': [
                {'range': [0, 30], 'color': COLORS['light_green']},
                {'range': [30, 60], 'color': COLORS['light_orange']},
                {'range': [60, 100], 'color': COLORS['light_blue']},
            ],
        }
    ))
    fig_gauge.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='white',
        plot_bgcolor='white',
    )

    return (
        kpi_cards,
        fig_top_domaines,
        fig_salaire_funnel,
        fig_offres_pays,
        fig_categories_pie,
        fig_salaire_pays,
        fig_offres_temps,
        fig_experience_donut,
        fig_salaire_experience,
        fig_villes,
        fig_contrats_pie,
        fig_salaire_pays_stacked,
        fig_gauge,
    )


if __name__ == "__main__":
    app.run(debug=True, port=8050)
