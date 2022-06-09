import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

@st.cache(allow_output_mutation=True)
def get_data(data):
    return pd.read_csv(data, delimiter=',')

df1 = get_data('RPL_games.csv')
df1['home'] = df1['games'].str[0]
df1['away'] = df1['games'].str[-1]

df_results = pd.pivot_table(data=df1, values='games', index='home', columns='away', aggfunc='count').fillna(0)


fig, ax = plt.subplots()
chart = sns.heatmap(data=df_results,
                     ax=ax, cbar=True, cmap='Oranges', annot=True, linewidths=.5)
chart.set_title(f"rgggg")
chart.set(xlabel='Home', ylabel='Guest')
st.pyplot(fig)

df2 = get_data("RPL_players.csv")

df3 = df2.loc[0:9].set_index('Игрок')
df3['С игры'] = df3['Голы'].astype(int) - df3['Пенальти'].astype(int)

df4 = pd.concat([df3['С игры'], df3['Пенальти'].astype(int)], ignore_index=False)
df4 = df4.reset_index()
df4.loc[0:9, 'Тип'] = 'С игры'
df4.loc[10:19, 'Тип'] = 'С пенальти'

fig, ax = plt.subplots()
chart = sns.barplot(x=0, y="Игрок", hue="Тип", data=df4)
chart.set_title(f"rgggg")
chart.set(xlabel='Количество голов', ylabel='Игрок')
st.pyplot(fig)

df2['Мин./Гол'] = df2['Мин./Гол'].astype(float)
df5 = df2.sort_values('Мин./Гол', ascending=True).iloc[0:14]

fig, ax = plt.subplots()
chart = sns.barplot(x='Мин./Гол', y="Игрок", data=df5, palette="Blues_d")
chart.set_title(f"rgggg")
chart.set(xlabel='Количество минут на гол', ylabel='Игрок')
st.pyplot(fig)

df6 = df2[['Национальность', 'Голы']].groupby('Национальность').count().drop('_country_flag', axis=0).reset_index()
df2['Голы'] = df2['Голы'].astype(int)
df7 = df2[['Национальность', 'Голы']].groupby('Национальность').sum().drop('_country_flag', axis=0).reset_index()

df8 = df6.merge(df7, left_on='Национальность', right_on='Национальность').rename(columns={'Голы_x': 'Количество футболистов', 'Голы_y': 'Суммарное количество голов'})

fig = px.pie(df8, values='Суммарное количество голов', names='Национальность', title=f'ggggggggg',
             color='Национальность')
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(showlegend=True)
fig.update_layout(font=dict(size=14))
st.plotly_chart(fig)

fig = px.pie(df8, values='Количество футболистов', names='Национальность', title=f'ggggggggg',
             color='Национальность')
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(showlegend=True)
fig.update_layout(font=dict(size=14))
st.plotly_chart(fig)





