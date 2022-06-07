import streamlit as st
import requests
import pandas as pd
import re
import folium

with st.echo(code_location='below'):

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
        list_for_address = re.findall("[\w\.]*\s", CL_2021_teams.loc[id, 'address'])
        new_address = list_for_address[0]
        for element in list_for_address:
            if element != new_address:
                new_address = new_address + str(element)
        if get_address(CL_2021_teams.loc[id, 'address']) != []:
            CL_2021_teams.loc[id, 'lat'] = float(get_address(CL_2021_teams.loc[id, 'address'])[0]['lat'])
            CL_2021_teams.loc[id, 'lon'] = float(get_address(CL_2021_teams.loc[id, 'address'])[0]['lon'])
        elif get_address(new_address) != []:
            CL_2021_teams.loc[id, 'lat'] = float(get_address(new_address)[0]['lat'])
            CL_2021_teams.loc[id, 'lon'] = float(get_address(new_address)[0]['lon'])
        elif get_address(CL_2021_teams.loc[id, 'venue']) != []:
            CL_2021_teams.loc[id, 'lat'] = float(get_address(CL_2021_teams.loc[id, 'venue'])[0]['lat'])
            CL_2021_teams.loc[id, 'lon'] = float(get_address(CL_2021_teams.loc[id, 'venue'])[0]['lon'])
        else:
            pass

    CL_2021_teams.loc[32, 'lat'], CL_2021_teams.loc[32, 'lon'] = 50.451531, 30.533692
    CL_2021_teams.loc[42, 'lat'], CL_2021_teams.loc[42, 'lon'] = 46.838346, 29.557169
    CL_2021_teams.loc[46, 'lat'], CL_2021_teams.loc[46, 'lon'] = 35.894864, 14.415372
    CL_2021_teams.loc[57, 'lat'], CL_2021_teams.loc[57, 'lon'] = 53.225338, -3.076505
    CL_2021_teams.loc[66, 'lat'], CL_2021_teams.loc[66, 'lon'] = 42.0186, 20.9783
    CL_2021_teams.loc[68, 'lat'], CL_2021_teams.loc[68, 'lon'] = 43.9746459, 12.5072119
    CL_2021_teams.loc[76, 'lat'], CL_2021_teams.loc[76, 'lon'] = 46.666667, 16.166667
    CL_2021_teams.loc[79, 'lat'], CL_2021_teams.loc[79, 'lon'] = 35.1725, 33.365

    m = folium.Map([55.75215, 37.61819], zoom_start=12)
    for ind, row in CL_2021_teams.iterrows():
        folium.Circle([row.lat, row.lon], radius=10).add_to(m)
    m




    areas
    competitions
    PL_2021_matches
    CL_2021_teams



    """ifjrhightigh"""
