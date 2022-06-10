import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
from sklearn.linear_model import LinearRegression

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

a = st.selectbox('Choose:', top_5_results['name'].unique())

fig, ax = plt.subplots()
chart = sns.barplot(y=top_5_results[lambda x: x['name']==a]['season'],
                    x=top_5_results[lambda x: x['name']==a]['total_home_goals'] +
                    top_5_results[lambda x: x['name']==a]['total_away_goals'], ax=ax, palette='flare')
chart.bar_label(chart.containers[0], fontsize=7, color='black')
chart.set_title(f'ggthyhyh')
chart.set_ylabel('Season')
chart.set_xlabel('Goals')
st.pyplot(fig)


selection = alt.selection_multi(fields=['name'], bind='legend')

alt_chart =  (alt.Chart(top_5_results).mark_line(point=True).encode(
    x=alt.X("season", scale=alt.Scale(zero=False), title="Season"),
    y=alt.Y("average_goals" , scale=alt.Scale(zero=False), axis=alt.Axis(title='Average goals')),
    color=alt.Color("name", legend=alt.Legend(title='The championships')),
    tooltip=[alt.Tooltip('name'), alt.Tooltip('season'), alt.Tooltip('average_goals')],
    opacity=alt.condition(selection, alt.value(1), alt.value(0.2))
).properties(
    title=f"The dynamics",
    width=700,
    height=700,
).add_selection(selection).interactive())

st.altair_chart(alt_chart)

### FROM (частично): Лекция за 24.05.
def add_jitter(x, jitter=0.4):
    return x+ np.random.uniform(low=-jitter, high=jitter, size=x.shape)
### END FROM

fig, ax = plt.subplots()
chart = sns.jointplot(data=top_5_matches.sort_values(['total', 'difference', 'season', 'name']).assign(home_team_goal=lambda x: add_jitter(x['home_team_goal']),
                                                away_team_goal=lambda x: add_jitter(x['away_team_goal'])),
                      x='home_team_goal', y='away_team_goal', ax=ax, hue="name", alpha=0.2, marginal_ticks=True)
st.pyplot(chart)

model = LinearRegression()
model.fit(top_5_matches[['total']], top_5_matches['difference'])

fig, ax = plt.subplots()
top_5_matches.plot.scatter(x="total", y="difference", ax=ax)
x = pd.DataFrame(dict(total=np.linspace(0, 12)))
plt.plot(x["total"], model.predict(x), color="C1", lw=2)
st.pyplot(fig)