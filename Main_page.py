import streamlit as st
import requests
import pandas as pd
import re
import folium
import networkx as nx
from pyvis.network import Network
from stvis import pv_static

with st.echo(code_location='below'):

    st.markdown("# Main page")
    st.sidebar.markdown("# Main page")

    @st.cache
    def get_api(url):
        return requests.get(url, headers={'X-Auth-Token': 'bc43b7291e5f4468ab7e843cfd0178a4'}, params={'format': 'json'}).json()

    areas = pd.DataFrame(get_api('http://api.football-data.org/v4/areas/')['areas'])
    competitions = pd.DataFrame(get_api('http://api.football-data.org/v4/competitions/')['competitions'])
    PL_2021_matches = pd.DataFrame(get_api('http://api.football-data.org/v4/competitions/2021/matches?dateFrom=2021-08-01&dateTo=2022-05-30')['matches'])
    CL_2021_teams = pd.DataFrame(get_api('http://api.football-data.org/v4/competitions/CL/teams?season=2021')['teams'])

    @st.cache
    def get_address(address):
        entrypoint = "https://nominatim.openstreetmap.org/search"
        params = {'q': address, 'format': 'json'}
        return requests.get(entrypoint, params=params).json()

    CL_2021_teams.loc[:, 'lat'] = 0.0
    CL_2021_teams.loc[:, 'lon'] = 0.0

    for id in CL_2021_teams.index:
        new_address = re.findall("[\w\.\-\s]*\s", CL_2021_teams.loc[id, 'address'])[0]
        if get_address(CL_2021_teams.loc[id, 'venue']) != []:
            CL_2021_teams.loc[id, 'lat'] = float(get_address(CL_2021_teams.loc[id, 'venue'])[0]['lat'])
            CL_2021_teams.loc[id, 'lon'] = float(get_address(CL_2021_teams.loc[id, 'venue'])[0]['lon'])
        elif get_address(CL_2021_teams.loc[id, 'address']) != []:
            CL_2021_teams.loc[id, 'lat'] = float(get_address(CL_2021_teams.loc[id, 'address'])[0]['lat'])
            CL_2021_teams.loc[id, 'lon'] = float(get_address(CL_2021_teams.loc[id, 'address'])[0]['lon'])
        elif get_address(new_address) != []:
            CL_2021_teams.loc[id, 'lat'] = float(get_address(new_address)[0]['lat'])
            CL_2021_teams.loc[id, 'lon'] = float(get_address(new_address)[0]['lon'])
        else:
            pass

    CL_2021_teams.loc[21, 'lat'], CL_2021_teams.loc[21, 'lon'] = 41.039206, 28.994742
    CL_2021_teams.loc[32, 'lat'], CL_2021_teams.loc[32, 'lon'] = 50.451531, 30.533692
    CL_2021_teams.loc[35, 'lat'], CL_2021_teams.loc[35, 'lon'] = 50.099803, 14.415911
    CL_2021_teams.loc[39, 'lat'], CL_2021_teams.loc[39, 'lon'] = 40.154605, 44.475231
    CL_2021_teams.loc[42, 'lat'], CL_2021_teams.loc[42, 'lon'] = 46.838346, 29.557169
    CL_2021_teams.loc[44, 'lat'], CL_2021_teams.loc[44, 'lon'] = 49.980858, 36.261703
    CL_2021_teams.loc[46, 'lat'], CL_2021_teams.loc[46, 'lon'] = 35.894864, 14.415372
    CL_2021_teams.loc[47, 'lat'], CL_2021_teams.loc[47, 'lon'] = 54.582625, -5.955189
    CL_2021_teams.loc[50, 'lat'], CL_2021_teams.loc[50, 'lon'] = 48.198036, 16.266021
    CL_2021_teams.loc[57, 'lat'], CL_2021_teams.loc[57, 'lon'] = 53.225338, -3.076505
    CL_2021_teams.loc[66, 'lat'], CL_2021_teams.loc[66, 'lon'] = 42.0186, 20.9783
    CL_2021_teams.loc[68, 'lat'], CL_2021_teams.loc[68, 'lon'] = 43.9746459, 12.5072119
    CL_2021_teams.loc[73, 'lat'], CL_2021_teams.loc[73, 'lon'] = 43.238331, 76.92435
    CL_2021_teams.loc[76, 'lat'], CL_2021_teams.loc[76, 'lon'] = 46.666667, 16.166667
    CL_2021_teams.loc[79, 'lat'], CL_2021_teams.loc[79, 'lon'] = 35.1725, 33.365

    m = folium.Map([55, 20], zoom_start=3.2, tiles='openstreetmap')
    for id in CL_2021_teams.index:
        folium.Circle([CL_2021_teams.loc[id, 'lat'], CL_2021_teams.loc[id, 'lon']],
                      radius=10, color='red', fill_color='Red').add_to(m)
        folium.Marker([CL_2021_teams.loc[id, 'lat'], CL_2021_teams.loc[id, 'lon']],
                      popup=f"{CL_2021_teams.loc[id, 'name']} \n ({CL_2021_teams.loc[id, 'venue']})",
                      icon=folium.Icon(color="red", icon="info-sign")).add_to(m)
    m

    EC_2018_matches = pd.DataFrame(get_api('http://api.football-data.org/v4/competitions/EC/matches?season=2021')['matches'])


    EC_2018_matches['Home'] = EC_2018_matches['homeTeam'].apply(lambda x: x['name'])
    EC_2018_matches['Away'] = EC_2018_matches['awayTeam'].apply(lambda x: x['name'])

    EC_2018_matches_part = EC_2018_matches[['Home', 'Away']]

    games_graph = nx.DiGraph([(home, away) for (home, away) in EC_2018_matches_part.values])

    net = Network(directed=False, notebook=True, height='700px', width='700px', bgcolor='#222222', font_color='white')
    net.from_nx(games_graph)

    net.repulsion(node_distance=100, central_gravity=0.33, spring_length=210, spring_strength=0.1, damping=1)

    pv_static(net)




    #areas
    #competitions
    #PL_2021_matches
    #CL_2021_teams


