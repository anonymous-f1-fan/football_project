import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

with st.echo(code_location='below'):
    st.header("Анализ Российской Премьер-Лиги")

    st.sidebar.header("Анализ Российской Премьер-Лиги")

    """Теперь предлагаю проанализировать Российскую Премьер-Лигу. Для этого я скачал некоторую информацию с сайта championat.com
    и предлагаю внимательнее посмотреть на неё."""

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

    st.subheader("Анализ игр")
    st.sidebar.subheader("Анализ игр")

    """В продолжение предыдущего раздела я предлагаю посмотреть, как российские клубы играли в прошлом сезоне, 
    то есть с каким счётом заканчивались матчи и как часто. Но визуализации ниже вы можете увидеть это."""

    fig, ax = plt.subplots()
    chart = sns.heatmap(data=df_results,
                         ax=ax, cbar=True, cmap='Oranges', annot=True, linewidths=.5)
    chart.set_title(f"Результаты матчей РПЛ 2021/2022")
    chart.set(xlabel='Голы домашней команды', ylabel='Голы гостевой команды')
    st.pyplot(fig)

    """Отсюда видно, что наиболее популярным счётом был счёт 1:1, а больше двух мячей за игру команды забивали очень редко."""

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

    st.subheader("Анализ игроков")
    st.sidebar.subheader("Анализ игроков")

    """Далее предлагаю посмотерть на 10 лучших бомабрдиров чемпионата России (то есть на тех, кто забил больше всего) 
    и на то, как распределилсь у этих игроков голы с игры с пенальти."""

    fig, ax = plt.subplots()
    chart = sns.barplot(x=0, y="Игрок", hue="Тип", data=df4, palette='rocket')
    chart.set_title(f"Лучшие бомбардиры РПЛ 2021/2022")
    chart.set(xlabel='Количество голов', ylabel='Игрок')
    st.pyplot(fig)

    df2['Мин./Гол'] = df2['Мин./Гол'].astype(float)
    df5 = df2.sort_values('Мин./Гол', ascending=True).iloc[0:14]

    """Только что мы увидели имена тех, кто забил больше всех в сезоне. 
    Но как мне кажется, было бы также интересно узнать, кто забивал чаще, то есть кому в среднем 
    требовалось меньше всего минут, чтобы забить гол. Поэтому предлагаю посмотреть и на следующую визуализацию."""

    fig, ax = plt.subplots()
    chart = sns.barplot(x='Мин./Гол', y="Игрок", data=df5, palette="flare")
    chart.set_title(f"Самые часто забивающие игроки РПЛ 2021/2022")
    chart.set(xlabel='Количество минут на гол', ylabel='Игрок')
    st.pyplot(fig)

    """Как Вы видите, часть игроков присутствует в обоих топах. Значит, они провели действительно неплохой сезон.
    Но часть игроков есть только во втором топе. Следовательно, они часто забивали, но мало. 
    Значит, эти игроки провели мало времени на поле за сезон. Возможно, что они провели в России лишь половину сезона, 
    а затем перебрались за границу (или наоборот приехали в Россию лишь зимой). 
    А, возможно, они мало играли из-за травм или из-за недоверия тренера."""

    st.subheader("Анализ национальностей")
    st.sidebar.subheader("Анализ национальностей")

    """Кроме того, в предыдущих визуализациях можно было заметить по фамилиям, что часть игроков, забиваших голы в РПЛ - иностранцы.
    Предлагаю при помощи двух следующих визуализаций посомтреть, игроки каких национальностей забивали и как голы распределилсь между ними."""

    """Сначала предлагаю посмотреть на диагарамму, где показано какая часть голов была забита игроками каждой национальности."""

    df6 = df2[['Национальность', 'Голы']].groupby('Национальность').count().reset_index()
    df2['Голы'] = df2['Голы'].astype(int)
    df7 = df2[['Национальность', 'Голы']].groupby('Национальность').sum().reset_index()

    df8 = df6.merge(df7, left_on='Национальность', right_on='Национальность').rename(columns={'Голы_x': 'Количество футболистов', 'Голы_y': 'Суммарное количество голов'})

    fig = px.pie(df8, values='Суммарное количество голов', names='Национальность', title=f'Распределение голов между национальностями',
                 color='Национальность')
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=True)
    fig.update_layout(font=dict(size=14))
    st.plotly_chart(fig)

    """А теперь предлагаю посмотреть на то, сколько игроков каждой национальности отличались в матчах РПЛ прошлого сезона."""

    fig = px.pie(df8, values='Количество футболистов', names='Национальность', title=f'Распределение бомбардиров между национальностями',
                 color='Национальность')
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=True)
    fig.update_layout(font=dict(size=14))
    st.plotly_chart(fig)

    """Из этих графиков видно, что подавляющее большинство голов было забито игроками из России (которых тоже больше, чем бомбардиров-иностранцев).
    Также можно заметить, что много голов забили бразильцы и колумбийцы. Кроме того, при дальнейшем анализе видно, например, что
    все голы Буркина-Фасо были забиты одним футболистом."""





