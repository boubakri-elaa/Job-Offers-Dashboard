import os
import pandas as pd

from dash import Dash, dcc, html, Input, Output
import plotly.express as px

# ========= chemins & données =========
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
ML_PATH = os.path.join(BASE_DIR, "data", "processed", "offres_ml.csv")

df = pd.read_csv(ML_PATH, encoding="utf-8")

villes_options = [{"label": v, "value": v} for v in sorted(df["Ville_propre"].dropna().unique())]
contrats_options = [{"label": c, "value": c} for c in sorted(df["Contrat_propre"].dropna().unique())]
clusters_options = [{"label": str(c), "value": int(c)} for c in sorted(df["cluster_metier"].unique())]

# couleurs claires & minimalistes
BG_COLOR = "#f5f0e8"      # beige très clair
CARD_COLOR = "#ffffff"    # blanc
ACCENT = "#2563eb"        # bleu
ACCENT2 = "#7c3aed"       # violet
ACCENT3 = "#16a34a"       # vert
TEXT_COLOR = "#1f2933"

app = Dash(__name__)

# Template HTML custom pour ajouter CSS hover
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Dashboard Offres Hellowork</title>
        {%css%}
        <style>
            .stat-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 10px rgba(15,23,42,0.18);
            }
            .card:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 14px rgba(15,23,42,0.18);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

# ========= layout =========
app.layout = html.Div(
    style={
        "fontFamily": "Segoe UI, Arial, sans-serif",
        "backgroundColor": BG_COLOR,
        "color": TEXT_COLOR,
        "minHeight": "100vh",
        "padding": "20px",
    },
    children=[
        # Titre dans un cadre
        html.Div(
            style={"display": "flex", "justifyContent": "center", "marginBottom": "18px"},
            children=[
                html.Div(
                    "Dashboard Offres Hellowork",
                    style={
                        "padding": "10px 24px",
                        "borderRadius": "14px",
                        "backgroundColor": CARD_COLOR,
                        "boxShadow": "0 2px 6px rgba(15,23,42,0.10)",
                        "fontSize": "26px",
                        "fontWeight": "700",
                        "color": ACCENT,
                    },
                )
            ],
        ),

        # Cartes de stats (centrées + hover)
        html.Div(
            style={
                "display": "flex",
                "gap": "12px",
                "marginBottom": "12px",
                "justifyContent": "center",
            },
            children=[
                html.Div(
                    style={
                        "minWidth": "180px",
                        "maxWidth": "200px",
                        "backgroundColor": CARD_COLOR,
                        "padding": "6px 10px",
                        "borderRadius": "10px",
                        "boxShadow": "0 1px 4px rgba(15,23,42,0.08)",
                        "transition": "transform 0.15s ease, box-shadow 0.15s ease",
                        "cursor": "default",
                    },
                    children=[
                        html.Div("Total offres", style={"fontSize": "11px", "color": "#6b7280"}),
                        html.Div(
                            f"{len(df):,}".replace(",", " "),
                            style={"fontSize": "18px", "fontWeight": "600", "color": ACCENT},
                        ),
                    ],
                    className="stat-card",
                ),
                html.Div(
                    style={
                        "minWidth": "180px",
                        "maxWidth": "200px",
                        "backgroundColor": CARD_COLOR,
                        "padding": "6px 10px",
                        "borderRadius": "10px",
                        "boxShadow": "0 1px 4px rgba(15,23,42,0.08)",
                        "transition": "transform 0.15s ease, box-shadow 0.15s ease",
                        "cursor": "default",
                    },
                    children=[
                        html.Div("Villes uniques", style={"fontSize": "11px", "color": "#6b7280"}),
                        html.Div(
                            f"{df['Ville_propre'].nunique()}",
                            style={"fontSize": "18px", "fontWeight": "600", "color": ACCENT2},
                        ),
                    ],
                    className="stat-card",
                ),
                html.Div(
                    style={
                        "minWidth": "210px",
                        "maxWidth": "230px",
                        "backgroundColor": CARD_COLOR,
                        "padding": "6px 10px",
                        "borderRadius": "10px",
                        "boxShadow": "0 1px 4px rgba(15,23,42,0.08)",
                        "transition": "transform 0.15s ease, box-shadow 0.15s ease",
                        "cursor": "default",
                    },
                    children=[
                        html.Div("% offres high demand", style={"fontSize": "11px", "color": "#6b7280"}),
                        html.Div(
                            f"{(df['pred_high_demand']==1).mean()*100:.1f} %",
                            style={"fontSize": "18px", "fontWeight": "600", "color": ACCENT3},
                        ),
                    ],
                    className="stat-card",
                ),
            ],
        ),

        # Filtres
        html.Div(
            style={"display": "flex", "gap": "12px", "marginBottom": "14px"},
            children=[
                html.Div(
                    style={"flex": 1},
                    children=[
                        html.Label("Ville", style={"fontSize": "12px", "color": "#4b5563"}),
                        dcc.Dropdown(
                            options=villes_options,
                            id="filter-ville",
                            placeholder="Toutes",
                            multi=True,
                        ),
                    ],
                ),
                html.Div(
                    style={"flex": 1},
                    children=[
                        html.Label("Contrat", style={"fontSize": "12px", "color": "#4b5563"}),
                        dcc.Dropdown(
                            options=contrats_options,
                            id="filter-contrat",
                            placeholder="Tous",
                            multi=True,
                        ),
                    ],
                ),
                html.Div(
                    style={"flex": 0.8},
                    children=[
                        html.Label("Cluster métier", style={"fontSize": "12px", "color": "#4b5563"}),
                        dcc.Dropdown(
                            options=clusters_options,
                            id="filter-cluster",
                            placeholder="Tous",
                            multi=True,
                        ),
                    ],
                ),
                html.Div(
                    style={"flex": 0.6},
                    children=[
                        html.Label("High demand", style={"fontSize": "12px", "color": "#4b5563"}),
                        dcc.Checklist(
                            options=[{"label": "Seulement", "value": 1}],
                            id="filter-high-demand",
                            value=[],
                            style={"fontSize": "12px"},
                        ),
                    ],
                ),
            ],
        ),

        # Graphiques ligne 1
        html.Div(
            style={"display": "flex", "gap": "16px", "marginBottom": "14px"},
            children=[
                html.Div(
                    className="card",
                    style={
                        "flex": 1,
                        "backgroundColor": CARD_COLOR,
                        "padding": "10px 12px",
                        "borderRadius": "10px",
                        "boxShadow": "0 1px 4px rgba(15,23,42,0.08)",
                        "transition": "transform 0.15s ease, box-shadow 0.15s ease",
                    },
                    children=[
                        html.H4("Offres par cluster", style={"marginBottom": "4px", "fontSize": "15px"}),
                        dcc.Graph(id="graph-cluster", style={"height": "300px"}),
                    ],
                ),
                html.Div(
                    className="card",
                    style={
                        "flex": 1,
                        "backgroundColor": CARD_COLOR,
                        "padding": "10px 12px",
                        "borderRadius": "10px",
                        "boxShadow": "0 1px 4px rgba(15,23,42,0.08)",
                        "transition": "transform 0.15s ease, box-shadow 0.15s ease",
                    },
                    children=[
                        html.H4("High demand vs autres", style={"marginBottom": "4px", "fontSize": "15px"}),
                        dcc.Graph(id="graph-high", style={"height": "300px"}),
                    ],
                ),
            ],
        ),

        # Graphiques ligne 2 : Top villes + pie clusters
        html.Div(
            style={"display": "flex", "gap": "16px"},
            children=[
                html.Div(
                    className="card",
                    style={
                        "flex": 1.2,
                        "backgroundColor": CARD_COLOR,
                        "padding": "10px 12px",
                        "borderRadius": "10px",
                        "boxShadow": "0 1px 4px rgba(15,23,42,0.08)",
                        "transition": "transform 0.15s ease, box-shadow 0.15s ease",
                    },
                    children=[
                        html.H4("Top 10 villes (nombre d'offres)", style={"marginBottom": "4px", "fontSize": "15px"}),
                        dcc.Graph(id="graph-ville", style={"height": "300px"}),
                    ],
                ),
                html.Div(
                    className="card",
                    style={
                        "flex": 0.8,
                        "backgroundColor": CARD_COLOR,
                        "padding": "10px 12px",
                        "borderRadius": "10px",
                        "boxShadow": "0 1px 4px rgba(15,23,42,0.08)",
                        "transition": "transform 0.15s ease, box-shadow 0.15s ease",
                    },
                    children=[
                        html.H4("Répartition des clusters", style={"marginBottom": "4px", "fontSize": "15px"}),
                        dcc.Graph(id="graph-pie", style={"height": "300px"}),
                    ],
                ),
            ],
        ),
    ],
)

# ========= filtrage =========
def apply_filters(df_in, villes, contrats, clusters, high_only):
    dff = df_in.copy()
    if villes:
        dff = dff[dff["Ville_propre"].isin(villes)]
    if contrats:
        dff = dff[dff["Contrat_propre"].isin(contrats)]
    if clusters:
        dff = dff[dff["cluster_metier"].isin(clusters)]
    if high_only:
        dff = dff[dff["pred_high_demand"] == 1]
    return dff

# ========= callbacks =========
@app.callback(
    Output("graph-cluster", "figure"),
    Output("graph-high", "figure"),
    Output("graph-ville", "figure"),
    Output("graph-pie", "figure"),
    Input("filter-ville", "value"),
    Input("filter-contrat", "value"),
    Input("filter-cluster", "value"),
    Input("filter-high-demand", "value"),
)
def update_graphs(villes, contrats, clusters, high_demand_value):
    high_only = 1 in (high_demand_value or [])
    dff = apply_filters(df, villes, contrats, clusters, high_only)

    # 1) cluster (bar)
    fig_cluster = px.histogram(
        dff,
        x="cluster_metier",
        color="cluster_metier",
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={"cluster_metier": "Cluster", "count": "Nombre d'offres"},
    )

    # 2) high demand vs autres (bar)
    counts = dff["pred_high_demand"].value_counts().rename_axis("type").reset_index(name="Nombre")
    mapping = {0: "Autres", 1: "High demand"}
    counts["type"] = counts["type"].map(mapping)
    fig_high = px.bar(
        counts,
        x="type",
        y="Nombre",
        color="type",
        color_discrete_map={"High demand": ACCENT3, "Autres": ACCENT},
        labels={"type": "", "Nombre": "Nombre d'offres"},
    )

    # 3) top villes (bar horizontal)
    top_villes = (
        dff["Ville_propre"]
        .value_counts()
        .head(10)
        .rename_axis("Ville")
        .reset_index(name="Nombre")
    )
    fig_ville = px.bar(
        top_villes,
        x="Nombre",
        y="Ville",
        orientation="h",
        color="Nombre",
        color_continuous_scale="Blues",
    )

    # 4) pie clusters
    pie_data = dff["cluster_metier"].value_counts().rename_axis("Cluster").reset_index(name="Nombre")
    fig_pie = px.pie(
        pie_data,
        names="Cluster",
        values="Nombre",
        hole=0.4,
        color="Cluster",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )

    for fig in (fig_cluster, fig_high, fig_ville, fig_pie):
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor=CARD_COLOR,
            plot_bgcolor=CARD_COLOR,
            font_color=TEXT_COLOR,
        )

    return fig_cluster, fig_high, fig_ville, fig_pie


if __name__ == "__main__":
    app.run(debug=True)
