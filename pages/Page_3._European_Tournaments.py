import streamlit as st
import requests
import pandas as pd
import re
import folium
import networkx as nx
from pyvis.network import Network
from stvis import pv_static
import matplotlib.pyplot as plt
import seaborn as sns

with st.echo(code_location='below'):

    st.header("Анализ европейских турниров")
    st.sidebar.markdown("# Анализ европейских турниров")

    """Теперь я предлагаю перейти от национальных турниров к турнирам, провоядщимся между команадами из разных стран."""

    st.subheader("Лига Чемпионов 2021/2022")

    """Первым таким турниром станет Лига Чемпионов - ежегодный турнир, в котором участвуют сильнейшие клубы из всех стран Европы. 
    Ниже я предлагаю выбрать одну из команд и посмотреть некоторые визуализации."""

    @st.cache
    def get_api(url, params={'format': 'json'}):
        return requests.get(url, headers={'X-Auth-Token': 'bc43b7291e5f4468ab7e843cfd0178a4'}, params=params).json()

    areas = pd.DataFrame(get_api('http://api.football-data.org/v4/areas/')['areas'])
    competitions = pd.DataFrame(get_api('http://api.football-data.org/v4/competitions/')['competitions'])
    PL_2021_matches = pd.DataFrame(get_api('http://api.football-data.org/v4/competitions/2021/matches',
                                           params={'format': 'json', 'dateFrom': '2021-08-01', 'dateTo': '2022-05-30'})['matches'])
    CL_2021_teams = pd.DataFrame(get_api('http://api.football-data.org/v4/competitions/CL/teams', params={'format': 'json', 'season': '2021'})['teams'])

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

    a = st.selectbox('Выберите:', CL_2021_teams['name'].unique())
    a_id = CL_2021_teams[lambda x: x['name'] == a].index[0]

    """На первой визуализации Вы можете увидеть, где находятся стадионы всех клубов, участвоваших в Лиге Чемпионов в сезоне 2021/2022.
    Эти стадионы показываются красными метками и при нажатии на метку можно посмотреть, какой клуб выступал на этом стадионе и как этот стадион
    называется. А выбранный ранее клуб показывается чёрной меткой."""

    m = folium.Map([55, 20], zoom_start=3.2, tiles='openstreetmap')

    for id in CL_2021_teams.index:
        folium.Circle([CL_2021_teams.loc[id, 'lat'], CL_2021_teams.loc[id, 'lon']],
                      radius=10, color='red', fill_color='Red').add_to(m)
        folium.Marker([CL_2021_teams.loc[id, 'lat'], CL_2021_teams.loc[id, 'lon']],
                      popup=f"{CL_2021_teams.loc[id, 'name']} \n ({CL_2021_teams.loc[id, 'venue']})",
                      icon=folium.Icon(color="red", icon="info-sign")).add_to(m)

    folium.Circle([CL_2021_teams.loc[a_id, 'lat'], CL_2021_teams.loc[a_id, 'lon']],
                  radius=10, color='black', fill_color='Black').add_to(m)
    folium.Marker([CL_2021_teams.loc[a_id, 'lat'], CL_2021_teams.loc[a_id, 'lon']],
                  popup=f"{CL_2021_teams.loc[a_id, 'name']} \n ({CL_2021_teams.loc[a_id, 'venue']})",
                  icon=folium.Icon(color="black", icon="info-sign")).add_to(m)

    m

    """Далее я предлагаю посмотреть на матчи, в которых принимала участие выбранная команда. 
    В предлагаемой таблице указаны участники матча (HomeTeam и AwayTeam) и то, сколько голов они забили (HomeGoals и AwayGoals соответсвенно)."""

    a_id2 = CL_2021_teams[lambda x: x['name'] == a]['id'].loc[a_id]

    a_matches = pd.DataFrame(get_api('http://api.football-data.org/v4/teams/'+str(a_id2)+'/matches/', params={'format': 'json', 'competitions': 'CL', 'season': '2021'})['matches'])
    a_matches1 = pd.DataFrame()
    a_matches1['HomeTeam'] = a_matches['homeTeam'].apply(lambda x: x['name'])
    a_matches1['AwayTeam'] = a_matches['awayTeam'].apply(lambda x: x['name'])
    a_matches1['HomeGoals'] = a_matches['score'].apply(lambda x: x['fullTime']['home'])
    a_matches1['AwayGoals'] = a_matches['score'].apply(lambda x: x['fullTime']['away'])
    a_matches1

    """Как мне кажется, эта таблица является довольно информативной, но, возможно, что не очень удобно смотреть то в один столбец, то в другой, чтобы понять, кто победил.
    Поэтому я предлагаю посмотреть на визуализацию, которая дополнит эту таблицу. На данной визуализации отмечена выбранная ранее команда и все команды, против которых она играла.
    Если линия зелёная, то это значит, что выбранная команда победила в этом матче. Если жёлтая, то была ничья.
    А если красная, то выбранная команда проиграла. Направление стрелки означает то, какая команада являлась домашней в этом матче. Если стрелка идёт от команды, то она играла домашний матч, 
    если она идёт к команде, то гостевой."""

    teams = list(set(a_matches1['HomeTeam'].unique()) | set(a_matches1['AwayTeam'].unique()))
    net = Network(directed=True, notebook=True, height='700px', width='700px', bgcolor='#222222', font_color='white')
    net.add_nodes(teams)

    for id in a_matches1.index:
        if a_matches1.loc[id, 'HomeTeam'] == a:
            if a_matches1.loc[id, 'HomeGoals'] > a_matches1.loc[id, 'AwayGoals']:
                net.add_edge(a_matches1.loc[id, 'HomeTeam'], a_matches1.loc[id, 'AwayTeam'], color='green')
            elif a_matches1.loc[id, 'HomeGoals'] < a_matches1.loc[id, 'AwayGoals']:
                net.add_edge(a_matches1.loc[id, 'HomeTeam'], a_matches1.loc[id, 'AwayTeam'], color='red')
            else:
                net.add_edge(a_matches1.loc[id, 'HomeTeam'], a_matches1.loc[id, 'AwayTeam'], color='yellow')
        if a_matches1.loc[id, 'AwayTeam'] == a:
            if a_matches1.loc[id, 'HomeGoals'] > a_matches1.loc[id, 'AwayGoals']:
                net.add_edge(a_matches1.loc[id, 'HomeTeam'], a_matches1.loc[id, 'AwayTeam'], color='red')
            elif a_matches1.loc[id, 'HomeGoals'] < a_matches1.loc[id, 'AwayGoals']:
                net.add_edge(a_matches1.loc[id, 'HomeTeam'], a_matches1.loc[id, 'AwayTeam'], color='green')
            else:
                net.add_edge(a_matches1.loc[id, 'HomeTeam'], a_matches1.loc[id, 'AwayTeam'], color='yellow')

    net.repulsion(node_distance=100, central_gravity=0.33, spring_length=210, spring_strength=0.1, damping=1)

    pv_static(net)

    st.subheader("Чемпионат Европы 2021")

    """Далее перейдём к Чемпионату Европы 2021 года. Этот турнир отличается от Лиги Чемпионов тем, что в нём участвуют 
    не кулбы из разных стран, а сборные самих стран."""

    """Обычно Чемпионат Европы проводится в одной-двух странах (но иногда бывают исключения), поэтому изображать 
    стадионы всех не имеет смысла (это не очень информативно). Однако посмотреть, кто с кем играл, - довольно интересно. 
    Поэтому ниже я предлагаю посмотреть на граф, на котором отмечены все страны-участницы ЧЕ-2021 и рёбрами отмечены страны, сыгравшие друг с другом."""

    EC_2021_matches = pd.DataFrame(get_api('http://api.football-data.org/v4/competitions/EC/matches', params={'format': 'json', 'season': '2021'})['matches'])

    EC_2021_matches['Home'] = EC_2021_matches['homeTeam'].apply(lambda x: x['name'])
    EC_2021_matches['Away'] = EC_2021_matches['awayTeam'].apply(lambda x: x['name'])

    EC_2021_matches_part = EC_2021_matches[['Home', 'Away']]

    games_graph = nx.DiGraph([(EC_2021_matches_part.loc[id, 'Home'], EC_2021_matches_part.loc[id, 'Away']) for id in EC_2021_matches_part.index])

    net = Network(directed=False, notebook=True, height='700px', width='700px', bgcolor='#222222', font_color='white')
    net.from_nx(games_graph)

    net.repulsion(node_distance=100, central_gravity=0.33, spring_length=210, spring_strength=0.1, damping=1)

    pv_static(net)

    EC_2021_scorers = pd.DataFrame(get_api('http://api.football-data.org/v4/competitions/EC/scorers',
                                           params={'format': 'json', 'season': '2021', 'limit': '15'})['scorers'])

    EC_2021_scorers['name'] = EC_2021_scorers['player'].apply(lambda x: x['name'])

    """Теперь немного прокомментирую предыдущую визуализацию. Так как Чемпионат Европы проводится в кубковом формате 
    (начиная с какого-то момента, команды играют на вылет), то большее число линий, идущих к одной стране, означает 
    более успешное выступление (команда сыграла больше матчей, а, значит, прошла дальше). Поэтому можно сделать вывод, что 
    самыми успешными командами были Италия и Англия и именно они дошли до финала."""

    """В заключение этого раздела предлагаю взглянуть на лучших бомбардиров Чемпионта Европы. """

    fig, ax = plt.subplots()
    chart = sns.barplot(x='goals', y="name", data=EC_2021_scorers, palette="Blues_d")
    chart.set_title(f"Лучшие бомбардиры Чемпионата Европы 2021")
    chart.set(xlabel='Количество голов', ylabel='Игрок')
    st.pyplot(fig)

    """При желании можно сравнить 
    этот график с соответствующим графиком для чемпионта России и сделать некоторые выводы. Например, можно заметить, 
    что в топе нет ни одного игрока, который был бы в топе РПЛ. К сожалению, это не значит, что игроки РПЛ слишком хороши 
    для ЧЕ и не участвуют в нём. Наоборот это значит, что чемпионат России не дотягивает до европейского уровня. 
    А также можно увидеть, что лучший бомбардир в РПЛ забил 19 голов, а в ЧЕ - лишь 5. Но стоит учитывать, что Гамид Агаларов 
    провёл 29 игр, а Патрик Шик всего лишь 5."""


