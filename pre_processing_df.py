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

df_structures.to_csv("/processed_df_structures.csv", index=False)
