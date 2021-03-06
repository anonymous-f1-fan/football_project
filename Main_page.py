import streamlit as st

st.header("Проект: Анализ данных, связанных с футболом")

st.subheader("Описание проекта")

"""Этот проект посвящён анализу и визуализации различных данных, связанных с мировым и российким футболом. Для удобства он
разделён на несколько частей, каждой из которых соответствует отдельная страничка, которую можно выбрать в верхнем левом углу."""

"""Первая часть проекта посвящена анализу топ-5 мировых чемпионатов: английская, испанская, итальянская, немецкая и французская лиги.
В этом разделе можно посмотреть информацию про каждый из чемпионатов по отдельности, сравнить их друг с другом и узнать про какие-то общие тенденции, 
характерные всем чемпионатам."""

"""Во второй части проекта я предлагаю взглянуть внимательнее на Российскую Премьер-Лигу. Я предлагаю расммотреть эту лигу 
отдельно, не сравнивая с топовыми чемпионатами, чтобы в лишний раз не разочаровываться в российском футболе."""

"""В третьей части рассматриваются европейские чемпионаты: Лига Чемпионов (турнир между лучшими европейскими клубами) и 
Чемпионат Европы (турнир между европейскими сборными). """

"""Заключительная часть посвящена анализу игроков и их рейтингов в игре FIFA 2020. Кроме того, для этой части был создан 
телеграм-бот, которым предлагается воспользоваться в конце этого раздела."""

st.subheader("Соответствие критериям")

"""При создании этого проекта использовалось множество различных технологий, поэтому для удобства проверяющих я опишу, 
какие технолгии использовались и для чего."""

st.markdown("**1. Обработка данных с помощью pandas.** Эту библиотеку я использовал повсеместно: создавал датафреймы, "
            "объединял, форматировал. Для этого использовались функции ```pivot_table```, ```groupby```, ```concat```, "
            "```astype```, ```merge```, ```apply``` и другие.")

st.markdown("**2. Веб-скреппинг.** Веб-скрэппинг использовался во второй части проекта, в которой я при помощи ```selenium``` "
            "получал информацию с сайта championat.com. Используемый код есть как в прикреплённом файле, так и в самом проекте.")

st.markdown("**3. Работа с REST API (XML/JSON).** Я работал с двумя разными API в третьей части проекта. "
            "Я получал информацию с api.football-data.org при помощи множества различных запросов "
            "(при этом этот API был с ограниченным доступом, поэтому я получал специальный доступ к базе данных, что, "
            "кажется, не обсуждалось в домашних заданиях). А затем я получал адреса "
            "стадионов, используя nominatim.openstreetmap.org.")

st.markdown('**4. Визуализация данных.** Визуализации есть во всех частях проекта. Кроме обычного ```matplotlib```, '
            'также использовались библиотеки ```seaborn```, ```altair``` и ```plotly```. Понятие сложности, конечно, субъективное, '
            'но от себя могу сказать, что, например, на построение "паутинок" в четвёртой части ушло довольно много времени и строчек кода.')

st.markdown('**5. Математические возможности Python.** Я использовал ```numpy``` в четырёх местах этого проекта: '
            'для создания шума в визуализации в первой части, при визуализации регрессий, при создании полярных координат для графика и при работе с данными из FIFA. '
            'При этом использовались ```np.linspace```, ```np.random.uniform```, ```np.concatenate``` и ```np.sum```.')

st.markdown('**6. Streamlit.** Кажется, получилось использовать.')

st.markdown('**7. SQL.** Использовал в первой части проекта. Так как используемая мной база данных была очень большой, то '
            'я сделал при помощи SQL несколько csv-файлов, с которыми дальше работал. Код можно найти в отдельном файле и в самом проекте.')

st.markdown('**8. Регулярные выражения.** Использовал дважды: когда скачивал информацию о национальностях футболистов с сайта championat.com и когда '
            'приводил адреса стадионов к стандартному виду (во многих адресах был указан почтовый индекс, из-за которого адрес стадиона не находился). '
            'Если бы регулярные выражения не были использованы в этих местах, то это сильно усложнило бы мне жизнь и пришлось бы писать много дополнительных строчек кода.')

st.markdown('**9. Работа с геоданными с помощью geopandas, shapely, folium и т.д.** Использовал ```shapely``` и ```folium```, когда изображал стадионы в третьей части.')

st.markdown('**10. Машинное обучение.** Использовал много различных регрессий в первой и последней частях проекта.')

st.markdown('**11. Работа с графами.** Изображал графы в третьей части проекта.')

st.markdown('**12. Дополнительные технологии.** Написал телеграм-бота.')

st.markdown('**13. Объём.** По-моему, тут около 900 строк. Точно больше 120.')

st.markdown('**14. Целостность проекта.** В проекте 4 части, но все они объединены темой "футбол".')

st.markdown('**15. Общее впечатление.** Надеюсь, мой проект понравится :)')







