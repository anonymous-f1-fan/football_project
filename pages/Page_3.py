import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

@st.cache(allow_output_mutation=True)
def get_data(data):
    return pd.read_csv(data, delimiter=',')

### Код, при помощи которого создаётся csv-файл
#driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
#driver.get("https://www.championat.com/football/_russiapl/tournament/4465/calendar/")

#new_list = []
#for i in range(240):
#    new_list.append(driver.find_elements(By.CSS_SELECTOR, "tr")[i+1].
#                    find_elements(By.TAG_NAME, "td")[-2].find_element(By.TAG_NAME, "span").
#                    get_attribute("innerHTML").strip())
#df = pd.DataFrame(new_list).rename(columns={0: 'games'})
#df.to_csv('RPL_games.csv', index=False)

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

### Код, при помощи которого создаётся csv-файл
#driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
#driver.get("https://www.championat.com/football/_russiapl/tournament/4465/statistic/player/bombardir/")

#new_list = []
#for i in range(196):
#    list_of_td = driver.find_elements(By.CSS_SELECTOR, "tr")[i+1].find_elements(By.CSS_SELECTOR, "td")
#    d = dict()
#    d[1] = re.findall('"[\w]*"', list_of_td[2].find_elements(By.CSS_SELECTOR, "span")[0].get_attribute("innerHTML").strip())[-1][1:-1]
#    d[2] = list_of_td[2].find_elements(By.CSS_SELECTOR, "span")[1].get_attribute("innerHTML").strip()
#    for j in range(3, 10):
#        d[j] = list_of_td[j].get_attribute("innerHTML").strip()
#    new_list.append(d)
#df = pd.DataFrame(new_list).rename(columns={1: "Национальность", 2: "Игрок", 3: "Клуб", 4: "Амплуа", 5: "Голы",
#                                            6: "Пенальти", 7: "Мин./Гол", 8: "Минуты", 9: "Игры"})

#df.to_csv("RPL_players.csv", index=False)


df2 = get_data("RPL_players.csv")

df3 = df2.loc[0:9].set_index('Игрок')
df3['С игры'] = df3['Голы'].astype(int) - df3['Пенальти'].astype(int)

df4 = pd.concat([df3['С игры'], df3['Пенальти'].astype(int)], ignore_index=False)
df4 = df4.reset_index()
df4.loc[0:9, 'Тип'] = 'С игры'
df4.loc[10:19, 'Тип'] = 'С пенальти'

fig, ax = plt.subplots()
chart = sns.barplot(x=0, y="Игрок", hue="Тип", data=df4, palette='rocket')
chart.set_title(f"rgggg")
chart.set(xlabel='Количество голов', ylabel='Игрок')
st.pyplot(fig)

df2['Мин./Гол'] = df2['Мин./Гол'].astype(float)
df5 = df2.sort_values('Мин./Гол', ascending=True).iloc[0:14]

fig, ax = plt.subplots()
chart = sns.barplot(x='Мин./Гол', y="Игрок", data=df5, palette="flare")
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





