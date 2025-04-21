#IMPORT LIBRARIES
import pandas as pd
import numpy as np
import pycountry # best for country level geocoding
import requests
import time
from googletrans import Translator
import networkx as nx
import plotly.graph_objects as go
import dash
from dash import html, dcc, Input, Output

# COLLECT DATA ----------------------------------------
df_structures = pd.read_csv('https://raw.githubusercontent.com/adeline-hub/feminine_kinship/refs/heads/main/Structures_femmes..csv')
df_structures.sample(2)

# PROCESS DATA ----------------------------------------

# Duplicate rows where 'Pays 2' is not empty
rows_to_duplicate = df_structures[df_structures['Pays 2'] != '']
duplicated_rows = rows_to_duplicate.copy()
# Replace values in 'Pays 1' of duplicated rows with values from 'Pays 2'
duplicated_rows['Pays'] = duplicated_rows['Pays 2']
# Append the duplicated rows to the original DataFrame
df_structures = pd.concat([df_structures, duplicated_rows], ignore_index=True)

# Duplicate rows where 'Pays 3' is not empty
rows_to_duplicate = df_structures[df_structures['Pays 3'] != '']
duplicated_rows = rows_to_duplicate.copy()
# Replace values in 'Pays 1' of duplicated rows with values from 'Pays 2'
duplicated_rows['Pays'] = duplicated_rows['Pays 3']
# Append the duplicated rows to the original DataFrame
df_structures = pd.concat([df_structures, duplicated_rows], ignore_index=True)

#df_structures.drop(columns=['Pays 2', 'Pays 3'], inplace=True)

# Delete rows where 'Pays 1' is null or NaN
df_structures = df_structures.dropna(subset=['Pays'])
# Additionally, if you want to remove empty strings as well, you can filter them out
df_structures = df_structures[df_structures['Pays'] != '']

# Corrected code using numpy.where
df_structures['link_Structure_membre'] = np.where(df_structures['Cout'] == 'Aucun', 'weak', 'strong')
df_structures['link_Structure_Benefices'] = np.where(df_structures['Composition'] == 'Femmes', 'women-centric', 'mixt')
df_structures['link_Structure_Stakeholders'] = np.where(df_structures['liens_Structure_Stakeholders'] == 'imposée', 'forced', 'free')

df_structures["Période_num"] = df_structures["Période historique"].str.extract(r"(\d{3,4})").astype(int)

# Initialize the translator
translator = Translator()

# Function to translate country names from French to English
def translate_country_name(country_name):
    try:
        translation = translator.translate(country_name, src='fr', dest='en')
        return translation.text
    except Exception as e:
        print(f"Error translating {country_name}: {e}")
        return None

# Apply the translation function to the 'Pays' column
df_structures['Country'] = df_structures['Pays'].apply(translate_country_name)

# Replace 'Middle East and North Africa' with 'Saudi Arabia'
df_structures['Country'] = df_structures['Country'].replace('Middle East and North Africa', 'Saudi Arabia')
df_structures['Country'] = df_structures['Country'].replace('West Africa', 'Nigeria')
df_structures['Country'] = df_structures['Country'].replace('Suede', 'Sweden')
df_structures['Country'] = df_structures['Country'].replace('Overall', 'Earth')
df_structures['Country'] = df_structures['Country'].replace('Ivory Coast', "Côte d'Ivoire")
df_structures['Country'] = df_structures['Country'].replace('Russia', "Russian Federation")
df_structures['Country'] = df_structures['Country'].replace('The Netherlands', "Netherlands")
df_structures['Country'] = df_structures['Country'].replace('Scandinavia', "Sweden")
df_structures['Country'] = df_structures['Country'].replace('Benign', "Benin")
df_structures['Country'] = df_structures['Country'].replace('Korea', "South Korea")
df_structures['Country'] = df_structures['Country'].replace('Europe', "Italy")
df_structures['Country'] = df_structures['Country'].replace('Earth', "Switzerland")

# Function to get ISO code from country name 
def get_iso_code(country_name):
    try:
        return pycountry.countries.lookup(country_name).alpha_2
    except LookupError:
        return None
# Apply the function to the 'Pays' column to create a new 'ISO Code' column
df_structures['ISO2'] = df_structures['Country'].apply(get_iso_code)

def iso2_to_iso3(iso2_code):
    try:
        return pycountry.countries.get(alpha_2=iso2_code.upper()).alpha_3
    except:
        return None  # Handle cases like invalid code
df_structures["ISO Code"] = df_structures["ISO2"].apply(iso2_to_iso3)
df_structures["ISO"] = df_structures["ISO Code"]

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
    node_colors = {'structure': '#ff3390', 'membre': '#33ffa2', 'benefice': '#a233ff', 'stakeholder': '#ff3c33'}
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
        node_traces.append(go.Scatter(
            x=node_x, y=node_y, mode='markers+text', text=labels,
            textposition="bottom center", hoverinfo='text',
            marker=dict(size=20, color=node_colors[node_type], symbol=node_symbols[node_type]),
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
        text="By nambo yang",
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
        colorscale=[[0, "#90ff33"], [1, "mediumslateblue"]],
        showscale=False,
        marker_line_color="white",
        hovertext=df_map["Nom"],
        hoverinfo="text"
    ))

    fig.update_layout(
        geo=dict(showframe=False, showcoastlines=False, projection_type="natural earth"),
        margin=dict(l=20, r=20, t=40, b=20),
        height=600,
        title="Carte des structures par pays",
        plot_bgcolor='oldlace',  # Background color for the plot area
        paper_bgcolor='oldlace'  # Background color for the entire figure
    )
    return fig


# DASH APP

app = dash.Dash(__name__)
app.title = "GreenCircle - Women Kinships"

app.layout = html.Div([
    html.H1(
        "Networks of Solidarity: Kinship Charts for Women's Structures 🌍",
        style={
            "textAlign": "center",
            "fontFamily": "Arial, sans-serif"
        }
    ),
    html.Div([
        dcc.Slider(
            id="period-slider",
            min=int(df_structures["Période_num"].min()),
            max=int(df_structures["Période_num"].max()),
            value=int(df_structures["Période_num"].min()),
            marks={int(year): str(int(year)) for year in sorted(df_structures["Période_num"].unique())},
            step=None,
            tooltip={"always_visible": True}
        )
    ], style={"width": "80%", "margin": "auto", "marginBottom": "30px"}),

    html.Div([
        dcc.Dropdown(
            id="region-dropdown",
            options=[{"label": str(r), "value": str(r)} for r in df_structures["Region"].unique()],
            value="Afrique",
            placeholder="Choisir une région",
            style={"width": "40%", "display": "inline-block", "marginRight": "20px"}
        ),
        dcc.Dropdown(
            id="structure-dropdown",
            placeholder="Choisir une structure",
            style={"width": "50%", "display": "inline-block"}
        ),
    ], style={"width": "90%", "margin": "auto", "marginBottom": "20px"}),
    html.Div([
        dcc.Graph(id="network-graph", style={"width": "49%", "display": "inline-block"}),
        dcc.Graph(id="map-graph", style={"width": "49%", "display": "inline-block"}),
    ]),
    html.Div(id="structure-info", style={"width": "80%", "margin": "auto", "padding": "20px", "backgroundColor": "oldlace", "borderRadius": "10px", "marginTop": "20px", "fontSize": "18px", "lineHeight": "1.5"})
])

# Populate structures based on region
@app.callback(
    Output("structure-dropdown", "options"),
    Output("structure-dropdown", "value"),
    Input("region-dropdown", "value")
)
def update_structure_options(selected_region):
    filtered = df_structures[df_structures["Region"] == selected_region]
    options = [{"label": str(nom), "value": str(nom)} for nom in filtered["Nom"]]
    default_value = options[0]["value"] if options else None
    return options, default_value

# Update graphs
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
    
    # Extract structure information
    structure_info = df_structures[df_structures["Nom"] == structure_name].iloc[0]
    info_text = f"📝 {structure_info['Nom']}'s governance is {structure_info['Gouvernance']}, membership is {structure_info['Composition']}, and main benefits are {structure_info['Benefices']}."
    
    return fig1, fig2, info_text

if __name__ == "__main__":
    app.run(debug=True)
