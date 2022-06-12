import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

with st.echo(code_location='below'):
    st.header("Анализ игроков из FIFA 20")
    st.sidebar.header("Анализ игроков из FIFA 20")

    """FIFA - ежегодно выпускаемый футбольный симулятор. Каждый год раработчики EA Sports приписывают различные характеристики тысячам игроков 
    и выпускают игру, в которой можно играть за различные команды или создать свою. Но сейчас нас будут интересовать именно рейтинги 
    футболистов и характеристики, которые разработчики выставляют всем игрокам."""

    @st.cache(allow_output_mutation=True)
    def get_data(data):
        return pd.read_csv(data, delimiter=',')

    df = get_data('players_20.csv')

    st.subheader("Анализ игроков и команд")
    st.sidebar.subheader("Анализ игроков и команд")

    """Основным показателем игрока является его суммарный рейтинг. Так как сранивать отдельных игроков по рейтингу не очень интересно, 
    я предлагаю сравнить разные команды по рейтингам их игркоков."""

    df1 = df[lambda x: x['club'].isin(['FC Barcelona', 'Real Madrid', 'Paris Saint-Germain', 'Juventus', 'Liverpool'])]
    fig, ax = plt.subplots()
    chart = sns.boxplot(y='club', x='overall', data=df1, ax=ax, palette='magma')
    chart.set_title(f'Distribution of ratings for top teams')
    chart.set_ylabel('Team')
    chart.set_xlabel('Rating')
    st.pyplot(fig)

    """Теперь прокомметирую предыдущую визуализацию. Эта визуализация показывает, как распределены рейтинги игроков в пяти выбранных команадах. 
    Чёрные вертикальные черты - это минимальный и максимальный рейтинги футболистов из соответствующей команды. 
    Закрашенный прямоугольник - это диапазон рейтингов, в котором находится 50% "средних" игроков (то есть не лучших и не худших). 
    А чёрная линия в середине прямоугольника - рейтинг медианного футболиста в этой команде."""

    """Тогда после моих комментариев можно понять, что, например, игрок с самым высоким рейтингом играет за Барселону, 
    а самый высокий рейтинг у медианного игрока в Ювентусе."""

    """Однако общий рейтинг не является единственным показателем уровня футболиста. На самом деле их очень много, но обычно выделяют ещё 
    шесть показателей: скорость, дриблинг, пасы, удары, защита и физические данные футболиста. Я предлагаю выбрать двух футболистов 
    и посмотреть на сравнение этих игроков по данным параметрам"""

    df0 = df[lambda x: x['player_positions'] != 'GK']

    a = st.selectbox('Выберите первого игрока:', df0['short_name'].unique())
    b = st.selectbox('Выберите второго игрока:', df0[lambda x: x['short_name'] != a]['short_name'].unique())

    df2 = df[lambda x: x['short_name'].isin([a, b])].reset_index()[['short_name', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']]

    # FROM (частично, то есть взята идея, но модифицирована для случая с несколькими строками в датафрейме и адаптирована для моей задачи):
    # https://stackoverflow.com/questions/52910187/how-to-make-a-polygon-radar-spider-chart-in-python
    angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    stats_1 = np.concatenate((df2.drop('short_name', axis=1).loc[0], [df2.loc[0, 'pace']]))
    stats_2 = np.concatenate((df2.drop('short_name', axis=1).loc[1], [df2.loc[1, 'pace']]))
    angles = np.concatenate((angles, [angles[0]]))
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, stats_1, 'o-', linewidth=2, label=a, color='red')
    ax.fill(angles, stats_1, alpha=0.25, color='red')
    ax.plot(angles, stats_2, 'o-', linewidth=2, label=b, color='blue')
    ax.fill(angles, stats_2, alpha=0.25, color='blue')
    ax.set_thetagrids(angles * 180/np.pi, ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic', 'pace'])
    plt.legend(bbox_to_anchor=(0.1, 0.1))
    ax.set_title(f"Comparison of {a} and {b}")
    ax.grid(True)
    # END FROM

    st.pyplot(fig)

    """Благодаря этой визуализации хорошо видно, кто из выбранных игроков лучше и по каким параметрам."""

    """Мы ещё вернёмся к анализу различных характеристик игроков и даже попробуем понять, как формируется общий рейтинг, но 
    сейчас я предлагаю сконцентрироваться немного на другом."""

    st.subheader("Проверка связи возраста и успешности выступлений футболистов")
    st.sidebar.subheader("Проверка связи возраста и успешности выступлений футболистов")

    """Во-первых, рейтинг показывает то, насколько хорошо выступает футболист. Принято считать, что все спортсмены много тренируются и 
    постоянно прогрессируют, достигая в какой-то момент пика своей карьеры. После этого, если они продолжают 
    участвовать в соревнованиях, то их результаты идут на спад и ухадшаются с каждым годом."""

    """Такое предположение кажется мне весьма логичным.
    Предлагаю проверить его построив регрессию (не линейную, а квадратичную)."""


    model = LinearRegression()
    quadratic = PolynomialFeatures(degree=2)
    X_quad = quadratic.fit_transform(df[['age']])

    model.fit(X_quad, df['overall'])

    fig, ax = plt.subplots()
    df.plot.scatter(x="age", y="overall", ax=ax, alpha=0.25, color='steelblue')
    x = pd.DataFrame(dict(total=np.linspace(15, 42)))
    plt.plot(x["total"], model.predict(quadratic.fit_transform(x)), color="red", lw=2)
    plt.xlabel('Age')
    plt.ylabel('Rating')
    plt.title('Quadratic regression on age')
    st.pyplot(fig)

    """Из этой визуализации видно, что действительно у футболистов обычно есть пик карьеры, который в среднем приходится на 30 лет."""

    """Предлагаю также убедиться в этом и другим способом. EA Sports для каждого игрока определяет его потенциальный рейтинг, 
    который, как предполагают разработчики, может быть у футболиста на пике карьеры. Поэтому я предполагаю посмотреть на 
    разность между потенциалом и общим рейтингом каждого игрока и ворастом. Чтобы подтвердить нашу теорию, эта 
    разность должна уменьшаться с каждым годом, а к 30 годам она должна равняться нулю."""

    """Для проверки гипотезы я построил полиномиальную регрессию (старшую степень многочлена я выбрал равной 4) и вот, что получилось:"""


    df5 = df[['age', 'overall', 'potential']]
    df5['delta'] = df5['potential'] - df5['overall']

    model = LinearRegression()
    quadratic = PolynomialFeatures(degree=4)
    X_quad = quadratic.fit_transform(df[['age']])

    model.fit(X_quad, df5['delta'])

    fig, ax = plt.subplots()
    df5.plot.scatter(x="age", y="delta", ax=ax, alpha=0.25, color='steelblue')
    x = pd.DataFrame(dict(total=np.linspace(15, 42)))
    plt.plot(x["total"], model.predict(quadratic.fit_transform(x)), color='red', lw=2)
    plt.ylim([-1, 26])
    plt.xlabel('Age')
    plt.ylabel('Difference')
    plt.title('Polynomial regression on age')
    st.pyplot(fig)

    """Как мы видим, разность между потенциалом и текущими способностями действительно падает с возрастом, а к 30 годам достигает 0. 
    Однако потом регрессия пердстказывает, что раность продолжит уменьшаться и станет отрицательной. На самом деле, как мы увидели ранее, так 
    и должно быть (футболисты стареют и начинают играть хуже). Но разработчики FIFA считают немного иначе: для старых игроков в качестве потенциального уровня 
    они указывают текущий уровень, а не тот, который был на пике. Поэтому разность между потенциалом и текущим рейтингом не может быть отрицательной в FIFA, хотя 
    в реальности она обычно отрицательная."""

    st.subheader("Анализ зависимости рейтинга от разных параметров")
    st.sidebar.subheader("Анализ зависимости рейтинга от разных параметров")

    """Теперь вернёмся к показатлеям игроков и общим рейтингам. Попробуем понять, как формируются рейтинги игроков."""

    """Первое предположение, которое кажется логичным: чем больше сумма всех параметров, тем выше общий рейтинг.
    В среднем это действительно так, в чём несложно убедиться построив регрессию."""

    new_list = df0.columns[44:72].to_list()
    new_list.append('overall')
    df3 = df0[new_list]
    df3['total'] = np.sum(df3, axis=1) - df3['overall']

    df4 = df3[['overall', 'total']]

    model = LinearRegression()
    model.fit(df4[['total']], df4['overall'])

    fig, ax = plt.subplots()
    df4.plot.scatter(x="total", y="overall", ax=ax, alpha=0.25, color='steelblue')
    x = pd.DataFrame(dict(total=np.linspace(1000, 2200)))
    plt.plot(x["total"], model.predict(x), color="red", lw=2)
    plt.xlabel('Total skills')
    plt.ylabel('Rating')
    plt.title('Linear regression on total skills')
    st.pyplot(fig)

    """Однако так же понятно и видно, что такая оценка оказывается очень неточной. Поэтому я предлагаю попробовать оценить общий 
    рейтинг не по сумме всех показателей, а по 6 основным показателям. Для этого нужно построить линейную регрессию по 6 переменным. 
    В этом случае у нас не будет одной линией, которая будет усреднять значение общего рейтинга по переменной, а будет целая область. 
    Причём можно изобразить эти области в осях общий рейтинг-переменная. Сделав это, получим 6 графиков, где голубым показаны реальные значения 
    для игроков, а розовым - предсказываемые."""

    model = LinearRegression()
    model.fit(df0[['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']], df0['overall'])

    fig, ax = plt.subplots(2, 3, sharey=True, figsize=(12, 8))
    df0.plot.scatter(x="pace", y="overall", ax=ax[0][0], color='steelblue')
    ax[0][0].plot(
        df0["pace"],
        model.predict(df0[['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']]),
        "o",
        color="salmon",
        alpha=0.2)

    df0.plot.scatter(x="shooting", y="overall", ax=ax[0][1], color='steelblue')
    ax[0][1].plot(
        df0["shooting"],
        model.predict(df0[['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']]),
        "o",
        color="salmon",
        alpha=0.2)

    df0.plot.scatter(x="passing", y="overall", ax=ax[0][2], color='steelblue')
    ax[0][2].plot(
        df0["passing"],
        model.predict(df0[['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']]),
        "o",
        color="salmon",
        alpha=0.2)

    df0.plot.scatter(x="dribbling", y="overall", ax=ax[1][0], color='steelblue')
    ax[1][0].plot(
        df0["dribbling"],
        model.predict(df0[['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']]),
        "o",
        color="salmon",
        alpha=0.2)

    df0.plot.scatter(x="defending", y="overall", ax=ax[1][1], color='steelblue')
    ax[1][1].plot(
        df0["defending"],
        model.predict(df0[['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']]),
        "o",
        color="salmon",
        alpha=0.2)

    df0.plot.scatter(x="physic", y="overall", ax=ax[1][2], color='steelblue')
    ax[1][2].plot(
        df0["physic"],
        model.predict(df0[['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']]),
        "o",
        color="salmon",
        alpha=0.2)

    st.pyplot(fig)

    """Однако даже из этих картинок не до конца понятно, каким будет рейтинг при заданных параметрах. Но для того, чтобы 
    разобраться с этим я (напишу) телеграм-бота, который будет предказывать рейтинг футболиста с заданными параметрами."""











