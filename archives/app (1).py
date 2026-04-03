# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import pycountry  # best for country level geocoding
import requests
import time
from googletrans import Translator
import networkx as nx
import plotly.graph_objects as go
import dash
from dash import html, dcc, Input, Output, State


# COLLECT DATA ----------------------------------------
df_structures = pd.read_csv(
    'https://raw.githubusercontent.com/adeline-hub/feminine_kinship/refs/heads/main/processed_df_structures.csv'
)

# FUNCTIONS - PRE-PROCESS to build the ----------------------------------------

# KINSHIP CHART WITH NETWORKX v2
def build_structure_graph(df, structure_name):
    # Filter the selected structure
    row = df[df['Nom'] == structure_name].iloc[0]

    G = nx.Graph()
    pos = {}

    # Main structure node
    structure_id = f"S_{structure_name}"
    G.add_node(structure_id, label=structure_name, type='structure')
    pos[structure_id] = (0, 0)

    # Add related nodes with fixed layout
    def add_related_nodes(node_id, related_items, relation_type, x_offset, y_offset):
        for j, item in enumerate(related_items):
            item_id = f"{relation_type[0]}_{j}"
            G.add_node(item_id, label=item, type=relation_type)
            pos[item_id] = (x_offset + j * 150, y_offset)
            G.add_edge(node_id, item_id, type=relation_type)

    add_related_nodes(structure_id, row['link_Structure_Benefices'].split(', '), 'membre', -300, -150)
    add_related_nodes(structure_id, row['Benefices'].split(', '), 'benefice', -150, 150)
    add_related_nodes(structure_id, row['Stakeholder'].split(', '), 'stakeholder', 150, -50)

    # Colors and symbols
    node_colors = {'structure': '#FF33FF', 'membre': '#33ffa2', 'benefice': '#a233ff', 'stakeholder': '#ff3c33'}
    node_symbols = {'structure': 'square', 'membre': 'circle', 'benefice': 'triangle-up', 'stakeholder': 'diamond'}

    # Edges
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color='gray'), hoverinfo='none')

    # Nodes
    node_traces = []
    for node_type in node_colors:
        nodes = [n for n, d in G.nodes(data=True) if d['type'] == node_type]
        node_x = [pos[n][0] for n in nodes]
        node_y = [pos[n][1] for n in nodes]
        labels = [G.nodes[n]['label'] for n in nodes]
# Example updated node trace inside build_structure_graph function

        node_traces.append(go.Scatter(
            x=node_x, y=node_y, mode='markers+text', text=labels,
            textposition="bottom center", hoverinfo='text',
            marker=dict(
                size=40,                # Bigger size (was 20)
                color=node_colors[node_type],
                symbol=node_symbols[node_type],
                opacity=0.5             # Semi-transparent (0 is fully transparent, 1 is opaque)
            ),
            name=node_type.capitalize()
        ))


    # Final figure
    fig = go.Figure(
        data=[edge_trace] + node_traces,
        layout=go.Layout(
            title=f"Kinship: {structure_name}",
            titlefont_size=18,
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
            plot_bgcolor='oldlace',
            paper_bgcolor='oldlace'
        )
    )

    fig.add_annotation(
        text="nambo yang | personal research",
        xref="paper", yref="paper",
        x=1, y=0,  # bottom-right corner
        showarrow=False,
        font=dict(size=12, color="gray"),
        align="right",
        xanchor="right", yanchor="bottom"
    )
    return fig


# Minimal world map
def plot_map(df, selected_structure):
    df_map = df.copy()
    df_map["selected"] = df_map["Nom"] == selected_structure
    fig = go.Figure()

    fig.add_trace(go.Choropleth(
        locations=df_map["ISO"].astype(str),
        z=df_map["selected"].astype(int),
        colorscale=[[0, "#33FFA2"], [1, "mediumslateblue"]],
        showscale=False,
        marker_line_color="white",
        hovertext=df_map["Nom"],
        hoverinfo="text"
    ))

    fig.update_layout(
        geo=dict(showframe=False, showcoastlines=False, projection_type="natural earth"),
        margin=dict(l=20, r=20, t=40, b=20),
        height=600,
        title="Kinships map",
        plot_bgcolor='oldlace',
        paper_bgcolor='oldlace'
    )
    return fig


# DASH APP

app = dash.Dash(
    __name__,
    server=True,
    title="GreenCircle - Women Kinships",
    assets_folder='assets',
    meta_tags=[
        {"name": "description", "content": "Scroll into the power of women’s kinships"},
        {"property": "og:title", "content": "Scroll into the power of women’s kinships"},
        {"property": "og:description", "content": "Mapping structures of solidarity, benefit, and membership"},
        {"property": "og:image", "content": "https://github.com/adeline-hub/feminine_kinship/blob/main/assets/logo%20big%20challenges.png?raw=true"},
        {"property": "og:type", "content": "website"},
        {"name": "twitter:card", "content": "summary_large_image"},
        {"name": "twitter:title", "content": "Scroll into the power of women’s kinships"},
        {"name": "twitter:image", "content": "/assets/share.png"},
    ]
)

server = app.server 


app.layout = html.Div(
    children=[
        html.H1(
            "Scroll into the power of women’s kinships",
            style={
                "textAlign": "center",
                "fontFamily": "Courier New",
                "color": "white"
            }
        ),

        # (Place your other Divs and components here, like dropdowns, graphs, etc.)


    html.Div([
        dcc.Dropdown(
            id="region-dropdown",
            options=[{"label": str(r), "value": str(r)} for r in df_structures["Region"].unique()],
            value="Afrique",
            placeholder="Filter by region",
            style={"width": "40%", "display": "inline-block", "marginRight": "20px"}
        ),
        # Structure dropdown disabled to avoid manual selection; we use navigation buttons instead
        dcc.Dropdown(
            id="structure-dropdown",
            options=[],
            value=None,
            clearable=False,
            disabled=True,
            style={"width": "50%", "display": "inline-block"}
        ),
    ], style={"width": "90%", "margin": "auto", "marginBottom": "20px"}),

    html.Div([
        html.Button("Previous", id="prev-btn", n_clicks=0, style={"marginRight": "10px"}),
        html.Button("Next", id="next-btn", n_clicks=0),
        html.Span(id="page-indicator", style={"paddingLeft": "20px", "fontWeight": "bold"})
    ], style={"width": "90%", "margin": "auto", "marginBottom": "20px", "textAlign": "center"}),

    dcc.Store(id="page-index", data=0),

    html.Div([
        dcc.Graph(id="network-graph", style={"width": "49%", "display": "inline-block"}),
        dcc.Graph(id="map-graph", style={"width": "49%", "display": "inline-block"}),
    ]),

    html.Div(id="structure-info", style={
        "width": "80%",
        "margin": "auto",
        "padding": "20px",
        "backgroundColor": "#33FFA2",
        "borderRadius": "10px",
        "marginTop": "20px",
        "fontFamily": "Courier New",
        "color": "#313130",
        "fontSize": "18px",
        "lineHeight": "1.5"
    })
]
    ,
    style={
        "backgroundColor": "#313130",
        "color": "white",
        "fontFamily": "Courier New"
    }
)


# Callback: update structure options & value based on region and page index
@app.callback(
    Output("structure-dropdown", "options"),
    Output("structure-dropdown", "value"),
    Output("page-indicator", "children"),
    Input("region-dropdown", "value"),
    Input("page-index", "data")
)
def update_structure_options(selected_region, page_index):
    filtered = df_structures[df_structures["Region"] == selected_region]
    options = [{"label": str(nom), "value": str(nom)} for nom in filtered["Nom"]]
    if len(filtered) == 0:
        return [], None, "No structures available"
    # Clamp page_index
    page_index = page_index % len(filtered)
    value = filtered.iloc[page_index]["Nom"]
    return options, value, f"Structure {page_index + 1} of {len(filtered)}"


# Callback: handle Previous/Next buttons to update page index
@app.callback(
    Output("page-index", "data"),
    Input("prev-btn", "n_clicks"),
    Input("next-btn", "n_clicks"),
    State("page-index", "data"),
    State("region-dropdown", "value")
)
def update_page_index(prev_clicks, next_clicks, current_index, selected_region):
    filtered = df_structures[df_structures["Region"] == selected_region]
    n = len(filtered)
    if n == 0:
        return 0

    ctx = dash.callback_context
    if not ctx.triggered:
        return current_index
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "next-btn":
        current_index = (current_index + 1) % n
    elif button_id == "prev-btn":
        current_index = (current_index - 1) % n
    return current_index


# Callback: update graphs and info based on structure
@app.callback(
    Output("network-graph", "figure"),
    Output("map-graph", "figure"),
    Output("structure-info", "children"),
    Input("structure-dropdown", "value")
)
def update_graphs(structure_name):
    if not structure_name:
        return go.Figure(), go.Figure(), ""

    fig1 = build_structure_graph(df_structures, structure_name)
    fig2 = plot_map(df_structures, structure_name)

    structure_info = df_structures[df_structures["Nom"] == structure_name].iloc[0]
    info_text = (
    f"📝 {structure_info['Nom']} is governed through {structure_info['Gouvernance']}, "
    f"its membership consists of {structure_info['Composition']}, and its main benefits include {structure_info['Benefices']}."

    )

    return fig1, fig2, info_text


if __name__ == "__main__":
    app.run(debug=True, port=8051)
