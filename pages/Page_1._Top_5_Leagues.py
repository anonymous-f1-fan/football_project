import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
from sklearn.linear_model import LinearRegression

with st.echo(code_location='below'):
    st.header("Анализ топ-5 европейских лиг")

    st.sidebar.markdown("# Анализ топ-5 европейских лиг")

    """В этой части проекта используются данные о разных лигах мира за 2008-2016 года."""

    """Сначала предлагаю ознакомиться со статистикой по каждому чемпионату отдельно, а именно: посмотреть, сколько всего голов забивалось в рассмтариваемом чемпионате за несколько сезонов."""

    @st.cache(allow_output_mutation=True)
    def get_data(data):
        return pd.read_csv(data, delimiter=',')

    ### Код, при помощи которого создаётся csv-файл
    #pd.read_sql("""SELECT name, season, SUM(home_team_goal) AS total_home_goals, SUM(away_team_goal) AS total_away_goals,
    #                        AVG(home_team_goal+away_team_goal) AS average_goals
    #                        FROM Match
    #                        LEFT JOIN League
    #                        ON Match.league_id = League.id
    #                        WHERE name in ("England Premier League", "France Ligue 1", "Germany 1. Bundesliga", "Italy Serie A", "Spain LIGA BBVA")
    #                        GROUP BY name, season;""", conn).to_csv("top_results.csv", index=False)

    top_5_results = get_data('top_results.csv')

    ### Код, при помощи которого создаётся csv-файл
    #pd.read_sql(f"""SELECT Match.id AS id, name, season, home_team_goal, away_team_goal, (home_team_goal-away_team_goal) AS difference,
    #(home_team_goal+away_team_goal) AS total
    #                        FROM Match
    #                        LEFT JOIN League
    #                        ON Match.league_id = League.id
    #                        WHERE name in ("England Premier League", "France Ligue 1",
    #                        "Germany 1. Bundesliga", "Italy Serie A", "Spain LIGA BBVA");""", conn).to_csv("top_matches.csv", index=False)

    top_5_matches = get_data('top_matches.csv')

    a = st.selectbox('Выберите чемпионат:', top_5_results['name'].unique())

    fig, ax = plt.subplots()
    chart = sns.barplot(y=top_5_results[lambda x: x['name']==a]['season'],
                        x=top_5_results[lambda x: x['name']==a]['total_home_goals'] +
                        top_5_results[lambda x: x['name']==a]['total_away_goals'], ax=ax, palette='flare')
    chart.bar_label(chart.containers[0], fontsize=7, color='black')
    chart.set_title(f'Statistics for {a}')
    chart.set_ylabel('Season')
    chart.set_xlabel('Goals')
    st.pyplot(fig)

    """Как Вы могли заметить, разным чемпионатам свойственна разная динамика: где-то количество голов уменьшается, 
    а где-то наоборот растёт с каждым сезоном. Кроме того, суммарное количество голов в разных чемпионатах довольно сильно отличается. 
    Это может происходить по разным причинам, одной из которых является разное количество игр в разных лигах. Но, как мне кажется, другим
    праметром может быть, что разные страны практикуют преимущественно разные тактики. Например, где-то больше играют в атакующий футбол, 
    а где-то акцентируется больше внимания на обороне. Поэтому я предлагаю сравнить, как менялось среднее количество забитых мячей за матч
    в разных чемпионатах. Для этого привожу следующую визуализацию."""


    selection = alt.selection_multi(fields=['name'], bind='legend')

    alt_chart =  (alt.Chart(top_5_results).mark_line(point=True).encode(
        x=alt.X("season", scale=alt.Scale(zero=False), title="Season"),
        y=alt.Y("average_goals" , scale=alt.Scale(zero=False), axis=alt.Axis(title='Average goals per game')),
        color=alt.Color("name", legend=alt.Legend(title='The leagues')),
        tooltip=[alt.Tooltip('name'), alt.Tooltip('season'), alt.Tooltip('average_goals')],
        opacity=alt.condition(selection, alt.value(1), alt.value(0.2))
    ).properties(
        title=f"The dynamics of the average goals per game for different countries",
        width=700,
        height=700,
    ).add_selection(selection).interactive())

    st.altair_chart(alt_chart)

    """Итак, из графика видно, что в Германии в среднем больше забивают, а во Франции - в среднем меньше. 
    Возможно, что в одном чемпионате собрались хорошие защитники, а в другом - нападающие. 
    Кроме того видно, что в большей части чемпионатов в сезоне 2014/2015 было забито меньше голов, чем в предыдущий и следующий сезоны.
    Возможно, это можно объяснить тем, что летом 2014 года проходил Чемпионат Мира, который забрал много сил у игроков 
    и они стали не так результативны в следующем сезоне."""

    """Теперь, когда мы посомтрели, сколько в среднем за матч забивается голов, я предлагаю посмотреть, как распределяются 
    голы между командами, которые играют дома (у себя на стадионе) и которые играют в гостях (на стадионе соперника).
    Для этого предлагаю посомтреть внимательнее на следующую визуализацию."""

    ### FROM (частично): Лекция за 24.05.
    def add_jitter(x, jitter=0.4):
        return x + np.random.uniform(low=-jitter, high=jitter, size=x.shape)
    ### END FROM

    top_5_matches['Home goals'] = add_jitter(top_5_matches['home_team_goal'])
    top_5_matches['Away goals'] = add_jitter(top_5_matches['away_team_goal'])
    top_5_matches['League'] = top_5_matches['name']

    fig, ax = plt.subplots()
    chart = sns.jointplot(data=top_5_matches.sort_values(['total', 'difference', 'season', 'name']),
                          x='Home goals', y='Away goals', ax=ax, hue="League", alpha=0.2, marginal_ticks=False)
    st.pyplot(chart)

    """Каждая точка на данном графике моответсвует отдельному матчу, то есть количеству забитых домашней и гостевой командой голов.
    Чтобы было лучше видно, сколько матчей завершалось с тем или иным счётом, я добавил немного колебаний, чтобы с каким-то 
    счётом располагались не в одной точке, а внутри небольшого квадрата вокруг этой точки.
    Тогда из этой визуализации видно, что обычно команды збивают за игру от нуля до двух голов. 
    Три и более голов, команды забивают довольно редко. 
    А матчи, в которых обе команды забивают много голов - огромная редкость для европейского футбола."""

    """Кроме того из маржинальных распределний кажется, что дома команды забивают в среднем больше голов, чем играя в гостях.
    Это соотносится с распространённым мнением, что дома команды выступают лучше из-за поддержки болельщиков и других факторов.
    Предлагаю проверить это утверждение при помощи линейной регресии."""

    """Я рассматривал две величины: суммарное количество голов за матч и разницу мячей в пользу домашней команды. 
    То есть если разница оказывалась положительной, то это означало победу домашней команды, если отрицательной - поражение. 
    Тогда, чтобы подтвердить гипотезу, то хотелось бы, чтобы после построения регрессии соответсвующая прямая из нуля пошла вверх, в 
    сторону положительной разницы мячей."""

    model = LinearRegression()
    model.fit(top_5_matches[['total']], top_5_matches['difference'])

    fig, ax = plt.subplots()
    top_5_matches.plot.scatter(x="total", y="difference", ax=ax, color='blue')
    x = pd.DataFrame(dict(total=np.linspace(0, 12)))
    plt.plot(x["total"], model.predict(x), color="red", lw=2)
    plt.title('Linear regression on total goals and difference per match')
    plt.xlabel('Total goals')
    plt.ylabel('Difference')
    st.pyplot(fig)

    """После построения регрессии мы получили именно то, что хотели: в среднем разниуа мячей положительная. 
    То есть в среднем дома команды выступают лучше, чем в гостях (по крайней мере в рассматриваемых чемпионатах в рассматриваемый период)."""