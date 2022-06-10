import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

@st.cache(allow_output_mutation=True)
def get_data(data):
    return pd.read_csv(data, delimiter=',')

df = get_data('players_20.csv')


df1 = df[lambda x: x['club'].isin(['FC Barcelona', 'Real Madrid', 'Paris Saint-Germain', 'Juventus', 'Liverpool'])]
fig, ax = plt.subplots()
chart = sns.boxplot(y='club', x='overall', data=df1, ax=ax, palette='flare')
st.pyplot(fig)

df0 = df[lambda x: x['player_positions'] != 'GK']

a = st.selectbox('Choose:', df0['short_name'].unique())
b = st.selectbox('Choose:', df0[lambda x: x['short_name'] != a]['short_name'].unique())

df2 = df[lambda x: x['short_name'].isin([a, b])].reset_index()[['short_name', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']]

# FROM (частично, то есть взята идея, но модифицирована для случая с несколькими строками в датафрейме и адаптирована для моей задачи):
# https://stackoverflow.com/questions/52910187/how-to-make-a-polygon-radar-spider-chart-in-python
angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
stats_1 = np.concatenate((df2.drop('short_name', axis=1).loc[0], [df2.loc[0, 'pace']]))
stats_2 = np.concatenate((df2.drop('short_name', axis=1).loc[1], [df2.loc[1, 'pace']]))
angles = np.concatenate((angles, [angles[0]]))
fig = plt.figure()
ax = fig.add_subplot(111, polar=True)
ax.plot(angles, stats_1, 'o-', linewidth=2, label=a)
ax.fill(angles, stats_1, alpha=0.25)
ax.plot(angles, stats_2, 'o-', linewidth=2, label=b)
ax.fill(angles, stats_2, alpha=0.25)
ax.set_thetagrids(angles * 180/np.pi, ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic', 'pace'])
plt.legend(bbox_to_anchor=(0.1, 0.1))
#plt.yticks(['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic', 'pace'])
#ax.set_title(name)
#ax.grid(True)
# END FROM

st.pyplot(fig)

new_list = df0.columns[44:72].to_list()
new_list.append('overall')
df3 = df0[new_list]
df3['total'] = np.sum(df3, axis=1) - df3['overall']

df4 = df3[['overall', 'total']]

model = LinearRegression()
model.fit(df4[['total']], df4['overall'])

fig, ax = plt.subplots()
df4.plot.scatter(x="total", y="overall", ax=ax, alpha=0.25)
x = pd.DataFrame(dict(total=np.linspace(1000, 2200)))
plt.plot(x["total"], model.predict(x), color="C1", lw=2)
st.pyplot(fig)



model = LinearRegression()
quadratic = PolynomialFeatures(degree=2)
X_quad = quadratic.fit_transform(df[['age']])

model.fit(X_quad, df['overall'])

fig, ax = plt.subplots()
df.plot.scatter(x="age", y="overall", ax=ax, alpha=0.25)
x = pd.DataFrame(dict(total=np.linspace(15, 42)))
plt.plot(x["total"], model.predict(quadratic.fit_transform(x)), color="C1", lw=2)
st.pyplot(fig)

df5 = df[['age', 'overall', 'potential']]
df5['delta'] =  df5['potential'] - df5['overall']




model = LinearRegression()
quadratic = PolynomialFeatures(degree=4)
X_quad = quadratic.fit_transform(df[['age']])

model.fit(X_quad, df5['delta'])

fig, ax = plt.subplots()
df5.plot.scatter(x="age", y="delta", ax=ax, alpha=0.25)
x = pd.DataFrame(dict(total=np.linspace(15, 42)))
plt.plot(x["total"], model.predict(quadratic.fit_transform(x)), color="C1", lw=2)
st.pyplot(fig)

model = LinearRegression()
model.fit(df0[['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']], df0['overall'])

fig, ax = plt.subplots(2, 3, sharey=True, figsize=(12, 8))
df0.plot.scatter(x="pace", y="overall", ax=ax[0][0])
ax[0][0].plot(
    df0["pace"],
    model.predict(df0[['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']]),
    "o",
    color="C1",
    alpha=0.2)

df0.plot.scatter(x="shooting", y="overall", ax=ax[0][1])
ax[0][1].plot(
    df0["shooting"],
    model.predict(df0[['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']]),
    "o",
    color="C1",
    alpha=0.2)

df0.plot.scatter(x="passing", y="overall", ax=ax[0][2])
ax[0][2].plot(
    df0["passing"],
    model.predict(df0[['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']]),
    "o",
    color="C1",
    alpha=0.2)

df0.plot.scatter(x="dribbling", y="overall", ax=ax[1][0])
ax[1][0].plot(
    df0["dribbling"],
    model.predict(df0[['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']]),
    "o",
    color="C1",
    alpha=0.2)

df0.plot.scatter(x="defending", y="overall", ax=ax[1][1])
ax[1][1].plot(
    df0["defending"],
    model.predict(df0[['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']]),
    "o",
    color="C1",
    alpha=0.2)

df0.plot.scatter(x="physic", y="overall", ax=ax[1][2])
ax[1][2].plot(
    df0["physic"],
    model.predict(df0[['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']]),
    "o",
    color="C1",
    alpha=0.2)

st.pyplot(fig)









