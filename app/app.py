import streamlit as st
import pandas as pd
from functions import calcular_distancia
from dateutil.relativedelta import relativedelta
import plotly.express as px
import os
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import pickle
from xgboost import XGBClassifier
from datetime import date
import joblib
from keras.models import load_model
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, accuracy_score

segmentos = []

data_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'data', 'raw'))

# Leer los archivos CSV segmentados y almacenarlos en la lista
for i in range(0,4):
    file_path = os.path.join(data_dir, f'segmento_{i+1}.csv')
    segmento = pd.read_csv(file_path)
    segmentos.append(segmento)
# Concatenar los DataFrames de los segmentos en uno solo
df1 = pd.concat(segmentos, ignore_index=True)

fraud_data = df1[df1['is_fraud'] == 1]  # Datos de fraudes

counts = df1['is_fraud'].value_counts()

# Encuentra la clase con el menor número de instancias
minority_class = counts.idxmin()

# Filtra el DataFrame para obtener solo instancias de la clase minoritaria
minority_df = df1[df1['is_fraud'] == minority_class]

# Obtiene una muestra aleatoria de la clase mayoritaria del mismo tamaño que la clase minoritaria
majority_df = df1[df1['is_fraud'] != minority_class].sample(n=len(minority_df), random_state=42)

# Combina los DataFrames de la clase minoritaria y la muestra de la clase mayoritaria
undersampled_df = pd.concat([minority_df, majority_df])

# Mezcla aleatoriamente las instancias en el DataFrame resultante
df_balanced= undersampled_df.sample(frac=1, random_state=42)


fraud_data = df1[df1['is_fraud'] == 1]  # Datos de fraudes


with open('../models/Model4/trained_model.pkl', 'rb') as f:
        modelo_cargado = pickle.load(f)
# Leer el archivo CSV de prueba
df = pd.read_csv("../data/test/test.csv")

# Obtener las características de prueba (X_test) y las etiquetas de prueba (y_test)
X_test = df[['amt', 'city_pop', 'distancia', 'fraudes_por_Categoria',
    'fraudes_por_estado', 'fraudes_por_edad', 'fraudes_por_hora',
    'fraudes_por_día']]
y_test = df['is_fraud']

# Interfaz de usuario con Streamlit
st.title("Detección de fraude con tarjeta de crédito")
st.image("image/Credit-Card-Fraud-investigation.jpg")

menu = st.sidebar.selectbox("Seleccionamos la página", ['Home','Análisis de datos exploratorio', 'Análisis procesamiento de datos', 'Modelo de predicción', 'Todos los modelos de predicción', 'Evaluación de modelos de predicción'])

if menu == "Home":
    st.markdown("En esta app, podrás probar un modelo de Machine Learning que ayudaría a dar solución en tiempo real a un problema del sector de la banca que va en aumento mientras la tecnología avanza a un ritmo frenético.")
    st.markdown("Introduciendo muy pocos datos de los clientes, casi todos existentes en cualquier base de datos de un banco, se puede predecir si una operación es fraudulenta o no, y por tanto, bloquear la transacción, ahorrando así costes del seguro de las tarjetas que da cobertura a los clientes en estos casos.")


if menu == "Análisis de datos exploratorio":
    st.markdown('Muestra del dataset con el que se ha trabajado')
    st.dataframe(df1.head())
    
    tab1, tab2 = st.tabs(['Ratio de fraudes','Correlación variables originales'])
    with tab1:
        fig = px.histogram(df1, x="is_fraud", color="is_fraud", color_discrete_map={0: "#34D399", 1: "#EF4444"},
                   labels={"is_fraud": "Transacción Fraudulenta"})

        # Actualizar las etiquetas de la leyenda
        fig.update_traces(
            name=["No Fraude", "Fraude"],
            legendtitle_text="Estado",
            patch={"line": {"color": "black", "width": 1}},
            selector={"legendgroup": True}
        )

        # Actualizar las etiquetas en el eje de color
        fig.for_each_trace(lambda t: t.update(name="No Fraude" if t.name == "0" else "Fraude"))

        # Configurar el título y las etiquetas de los ejes
        fig.update_layout(
            title_text="Distribución de Transacciones Fraudulentas",
            xaxis_title="Transacción Fraudulenta",
            yaxis_title="Conteo"
        )

        # Mostrar la gráfica
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        plt.figure(figsize=(15,10))
        sns.heatmap(df1.corr(numeric_only=True), annot=True)
        fig = plt.gcf()  # Obtener la figura actual
        st.pyplot(fig)
    
    tab3, tab4, tab5 = st.tabs(['Cuantía de las operaciones fraudulentas','Domicilio clientes que cometen fraudes','Comercios en los que se cometen fraudes'])
    with tab3:
        # Crear el histograma interactivo con Plotly
        fig = px.histogram(fraud_data, x="amt", color_discrete_sequence=["red"],
                        labels={"amt": "Cuantía de la transacción", "count": "Frecuencia"})

        # Configurar el diseño del gráfico
        fig.update_layout(title_text="Distribución de la cuantía de las transacciones fraudulentas",
                        xaxis_title="Cuantía de la transacción",
                        yaxis_title="Frecuencia")

        # Mostrar la gráfica en Streamlit
        st.plotly_chart(fig)

    with tab4:
        # Crea un objeto de mapa centrado en una ubicación inicial
        fraud_map = folium.Map(location=[fraud_data['lat'].mean(), fraud_data['long'].mean()], zoom_start=10)

        # Crea un grupo de marcadores para las ubicaciones de fraude
        fraud_markers = MarkerCluster().add_to(fraud_map)

        # Agrega marcadores para cada ubicación de fraude al grupo de marcadores
        for index, row in fraud_data.iterrows():
            folium.Marker([row['lat'], row['long']]).add_to(fraud_markers)

        # Crea una leyenda para el mapa
        legend_html = '''
            <div style="position: fixed; 
                        bottom: 50px; left: 50px; width: 120px; height: 90px; 
                        border:2px solid grey; z-index:9999; font-size:14px;
                        background-color: white;
                        ">
            <p style="margin: 5px;">Domicilios de clientes que cometen fraude</p>
            </div>
            '''

        # Agrega la leyenda al mapa
        fraud_map.get_root().html.add_child(folium.Element(legend_html))

        # Mostrar el mapa en Streamlit usando folium_static
        folium_static(fraud_map)

    with tab5:
        # Crea un mapa para ubicaciones de comercios
        merchant_map = folium.Map(location=[fraud_data['merch_lat'].mean(), fraud_data['merch_long'].mean()], zoom_start=10)

        # Crea un grupo de marcadores para las ubicaciones de comercios
        merchant_markers = MarkerCluster().add_to(merchant_map)

        # Agrega marcadores para cada ubicación de comercio al grupo de marcadores
        for index, row in fraud_data.iterrows():
            folium.Marker([row['merch_lat'], row['merch_long']]).add_to(merchant_markers)

        # Crea una leyenda para el mapa
        legend_html = '''
            <div style="position: fixed; 
                        bottom: 50px; left: 50px; width: 120px; height: 90px; 
                        border:2px solid grey; z-index:9999; font-size:14px;
                        background-color: white;
                        ">
            <p style="margin: 5px;">Comercios de fraudes</p>
            </div>
            '''

        # Agrega la leyenda al mapa
        merchant_map.get_root().html.add_child(folium.Element(legend_html))

        # Mostrar el mapa en Streamlit usando folium_static
        folium_static(merchant_map)

    tab6, tab7, tab8 = st.tabs(['Distribución fraudes por categoría de comercio','Distribución de fraudes por género','Distribución de fraude por estado'])
    with tab6:
    # Crear la gráfica dinámica con plotly
        fig = px.histogram(df_balanced, x="category", color="is_fraud", barmode="group",
                            color_discrete_map={0: "#34D399", 1: "#EF4444"},
                            labels={"category": "Categoría", "count": "Conteo", "is_fraud": "Fraude"})

        # Configurar el tamaño del gráfico y las propiedades de diseño
        fig.update_layout(
            title_text="Distribución de Categorías de Transacciones",
            xaxis_title="Categoría",
            yaxis_title="Conteo",
            bargap=0.1,  # Espacio entre las barras
            autosize=False,
            width=800,   # Ancho del gráfico
            height=500   # Altura del gráfico
        )

        # Rotar las etiquetas del eje x para que sean legibles
        fig.update_layout(xaxis={'tickangle': 45})

        st.plotly_chart(fig, use_container_width=True)
    with tab7:
        # Crear la gráfica dinámica con plotly
        fig = px.histogram(df_balanced, x="gender", color="is_fraud", barmode="group", color_discrete_map={0: "#34D399", 1: "#EF4444"},
                        labels={"gender": "Género", "is_fraud": "Transacción Fraudulenta"})

        # Actualizar las etiquetas de la leyenda
        fig.update_traces(
            name=["No Fraude", "Fraude"],
            legendtitle_text="Estado",
            patch={"line": {"color": "black", "width": 1}},
            selector={"legendgroup": True}
        )

        # Actualizar las etiquetas en el eje de color
        fig.for_each_trace(lambda t: t.update(name="No Fraude" if t.name == "0" else "Fraude"))

        # Configurar el título y las etiquetas de los ejes
        fig.update_layout(
            title_text="Distribución de Transacciones Fraudulentas por Género",
            xaxis_title="Género",
            yaxis_title="Conteo"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab8:
        # Crear la gráfica dinámica con plotly
        fig = px.histogram(df_balanced, x="state", color="is_fraud", barmode="group", color_discrete_map={0: "#34D399", 1: "#EF4444"},
                        labels={"state": "Estado", "is_fraud": "Transacción Fraudulenta"})

        # Actualizar las etiquetas de la leyenda
        fig.update_traces(
            name=["No Fraude", "Fraude"],
            legendtitle_text="Estado",
            patch={"line": {"color": "black", "width": 1}},
            selector={"legendgroup": True}
        )

        # Actualizar las etiquetas en el eje de color
        fig.for_each_trace(lambda t: t.update(name="No Fraude" if t.name == "0" else "Fraude"))

        # Configurar el título y las etiquetas de los ejes
        fig.update_layout(
            title_text="Distribución de Transacciones Fraudulentas por Estado",
            xaxis_title="Estado",
            yaxis_title="Conteo"
        )
        st.plotly_chart(fig, use_container_width=True)

if menu == 'Análisis procesamiento de datos':
    tab9, tab10, tab11 = st.tabs(['Distribución fraudes por edad','Distribución de hora','Distribución de fraude por día de la semana'])
    df1["dob"] = pd.to_datetime(df1["dob"])  # Convertir la columna "dob" a tipo de dato datetime
    df1["trans_date_trans_time"] = pd.to_datetime(df1["trans_date_trans_time"])  # Convertir la columna "trans_date_trans_time" a tipo de dato datetime
    df1["age"] = df1.apply(lambda row: relativedelta(row["trans_date_trans_time"], row["dob"]).years, axis=1)
    counts = df1['is_fraud'].value_counts()

    # Encuentra la clase con el menor número de instancias
    minority_class = counts.idxmin()

    # Filtra el DataFrame para obtener solo instancias de la clase minoritaria
    minority_df = df1[df1['is_fraud'] == minority_class]

    # Obtiene una muestra aleatoria de la clase mayoritaria del mismo tamaño que la clase minoritaria
    majority_df = df1[df1['is_fraud'] != minority_class].sample(n=len(minority_df), random_state=42)

    # Combina los DataFrames de la clase minoritaria y la muestra de la clase mayoritaria
    undersampled_df = pd.concat([minority_df, majority_df])

    # Mezcla aleatoriamente las instancias en el DataFrame resultante
    df_balanced= undersampled_df.sample(frac=1, random_state=42)

    with tab9:
        # Crear la gráfica utilizando Plotly Express
        fig = px.histogram(df_balanced, x="age", color="is_fraud", barmode="group", color_discrete_map={0: "#34D399", 1: "#EF4444"},
                        labels={"age": "Edad", "is_fraud": "Transacción Fraudulenta"})

        # Actualizar las etiquetas de la leyenda
        fig.update_traces(
            name=["No Fraude", "Fraude"],
            legendtitle_text="Estado",
            selector={"legendgroup": True}
        )

        fig.for_each_trace(lambda t: t.update(name="No Fraude" if t.name == "0" else "Fraude"))

        # Configurar el título y las etiquetas de los ejes
        fig.update_layout(
            title_text="Distribución de Transacciones Fraudulentas por Edad",
            xaxis_title="Edad",
            yaxis_title="Conteo"
        )

        # Mostrar la gráfica utilizando st.plotly_chart
        st.plotly_chart(fig, use_container_width=True)

    # Convertir la columna "trans_date_trans_time" a tipo de dato datetime
    df1["trans_date_trans_time"] = pd.to_datetime(df1["trans_date_trans_time"])

    # Generar columna con la hora (del 0 al 23) de la transacción
    df1["hour"] = df1["trans_date_trans_time"].dt.hour

    # Generar columna con el día de la semana (del 1 al 7) de la transacción
    df1["day_of_week"] = df1["trans_date_trans_time"].dt.dayofweek + 1

    # Generar columna con el día del mes (del 1 al 31) de la transacción
    df1["day_of_month"] = df1["trans_date_trans_time"].dt.day

    counts = df1['is_fraud'].value_counts()

    # Encuentra la clase con el menor número de instancias
    minority_class = counts.idxmin()

    # Filtra el DataFrame para obtener solo instancias de la clase minoritaria
    minority_df = df1[df1['is_fraud'] == minority_class]

    # Obtiene una muestra aleatoria de la clase mayoritaria del mismo tamaño que la clase minoritaria
    majority_df = df1[df1['is_fraud'] != minority_class].sample(n=len(minority_df), random_state=42)

    # Combina los DataFrames de la clase minoritaria y la muestra de la clase mayoritaria
    undersampled_df = pd.concat([minority_df, majority_df])

    # Mezcla aleatoriamente las instancias en el DataFrame resultante
    df_balanced= undersampled_df.sample(frac=1, random_state=42)

    with tab10:
        # Crear la gráfica utilizando Plotly Express
        fig = px.histogram(df_balanced, x="hour", color="is_fraud", barmode="group", color_discrete_map={0: "#34D399", 1: "#EF4444"},
                        labels={"hour": "Hora", "is_fraud": "Transacción Fraudulenta"})

        # Actualizar las etiquetas de la leyenda
        fig.update_traces(
            name=["No Fraude", "Fraude"],
            legendtitle_text="Estado",
            selector={"legendgroup": True}
        )

        fig.for_each_trace(lambda t: t.update(name="No Fraude" if t.name == "0" else "Fraude"))

        # Configurar el título y las etiquetas de los ejes
        fig.update_layout(
            title_text="Distribución de Transacciones Fraudulentas por Hora",
            xaxis_title="Hora",
            yaxis_title="Conteo"
        )

        # Mostrar la gráfica utilizando st.plotly_chart
        st.plotly_chart(fig, use_container_width=True)

    with tab11:
        # Crear la gráfica utilizando Plotly Express
        fig = px.histogram(df_balanced, x="day_of_month", color="is_fraud", barmode="group", color_discrete_map={0: "#34D399", 1: "#EF4444"},
                        labels={"day_of_month": "Día del mes", "is_fraud": "Transacción Fraudulenta"})

        # Actualizar las etiquetas de la leyenda
        fig.update_traces(
            name=["No Fraude", "Fraude"],
            legendtitle_text="Estado",
            selector={"legendgroup": True}
        )

        fig.for_each_trace(lambda t: t.update(name="No Fraude" if t.name == "0" else "Fraude"))

        # Configurar el título y las etiquetas de los ejes
        fig.update_layout(
            title_text="Distribución de Transacciones Fraudulentas por Día del mes",
            xaxis_title="Día del mes",
            yaxis_title="Conteo"
        )

        # Mostrar la gráfica utilizando st.plotly_chart
        st.plotly_chart(fig, use_container_width=True)

    tab12, tab13= st.tabs(['Nuevas correlaciones no balanceado','Nuevas correlaciones balanceado'])

    segmentos = []
    data_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'data', 'processed'))

    # Leer los archivos CSV segmentados y almacenarlos en la lista
    for i in range(0,3):
        file_path = os.path.join(data_dir, f'segmento_{i+1}.csv')
        segmento = pd.read_csv(file_path)
        segmentos.append(segmento)
    # Concatenar los DataFrames de los segmentos en uno solo
    df2 = pd.concat(segmentos, ignore_index=True)

    counts = df2['is_fraud'].value_counts()

    # Encuentra la clase con el menor número de instancias
    minority_class = counts.idxmin()

    # Filtra el DataFrame para obtener solo instancias de la clase minoritaria
    minority_df = df2[df2['is_fraud'] == minority_class]

    # Obtiene una muestra aleatoria de la clase mayoritaria del mismo tamaño que la clase minoritaria
    majority_df = df2[df2['is_fraud'] != minority_class].sample(n=len(minority_df), random_state=42)

    # Combina los DataFrames de la clase minoritaria y la muestra de la clase mayoritaria
    undersampled_df = pd.concat([minority_df, majority_df])

    # Mezcla aleatoriamente las instancias en el DataFrame resultante
    df_balanced2= undersampled_df.sample(frac=1, random_state=42)

    with tab12:
        plt.figure(figsize=(10,10))
        sns.heatmap(df2.corr(numeric_only=True), annot=True)
        fig = plt.gcf()  # Obtener la figura actual
        st.pyplot(fig)
    with tab13:
        plt.figure(figsize=(10,10))
        sns.heatmap(df_balanced2.corr(numeric_only=True), annot=True)
        fig = plt.gcf()  # Obtener la figura actual
        st.pyplot(fig)

if menu == "Modelo de predicción":
    st.text('Modelo de predicción con mejores resultados (XGBoostClassifier)')
    # Recoger las características de entrada del usuario
    amt = st.slider("Cuantía de la transacción", 1, 30000, 50)
    city_pop = st.slider("Población de la ciudad", 20, 3000000)
    lat = st.slider("Latitud del domicilio cliente", 19.02779, 67.51027, 0.00001)
    long = st.slider("Longitud del domicilio cliente", -166.6712, -66.9509, 0.00001)
    lat_merch = st.slider("Latitud del comercio", 19.02779, 67.51027, 0.00001)
    long_merch = st.slider("Longitud del comercio", -166.6712, -66.9509, 0.00001)
    distancia = calcular_distancia(lat, long, lat_merch, long_merch)

    categoria_options = ('misc_net', 'grocery_pos', 'entertainment', 'gas_transport',
                         'misc_pos', 'grocery_net', 'shopping_net', 'shopping_pos',
                         'food_dining', 'personal_care', 'health_fitness', 'travel',
                         'kids_pets', 'home')
    categoria = st.selectbox("Seleccione la categoría del comercio:", categoria_options)

    cat = pd.read_csv("categoria.csv")
    fraudes_por_Categoria = cat.loc[cat['category'] == categoria, 'fraudes_por_Categoria'].iloc[0]

    state_options = ('NC', 'WA', 'ID', 'MT', 'VA', 'PA', 'KS', 'TN', 'IA', 'WV', 'FL',
                     'CA', 'NM', 'NJ', 'OK', 'IN', 'MA', 'TX', 'WI', 'MI', 'WY', 'HI',
                     'NE', 'OR', 'LA', 'DC', 'KY', 'NY', 'MS', 'UT', 'AL', 'AR', 'MD',
                     'GA', 'ME', 'AZ', 'MN', 'OH', 'CO', 'VT', 'MO', 'SC', 'NV', 'IL',
                     'NH', 'SD', 'AK', 'ND', 'CT', 'RI', 'DE')
    estado = st.selectbox("Seleccione el estado:", state_options)

    state = pd.read_csv("estado.csv")
    fraudes_por_estado = state.loc[state['state'] == estado, 'fraudes_por_estado'].iloc[0]

    # Obtener la fecha y hora de transacción bancaria y la fecha de nacimiento
    trans_date = st.date_input("Fecha Transacción Bancaria")
    # Set the default start date to "01/01/1920"
    default_start_date = date(1920, 1, 1)

    # Create the date input widget
    dob = st.date_input("Fecha de Nacimiento", min_value=default_start_date)
    age = relativedelta(trans_date, dob).years
    
    edad = pd.read_csv("edad.csv")
    fraudes_por_edad = 0  # Valor predeterminado si no se encuentra una coincidencia

    if int(age) in edad['age'].values:
        fraudes_por_edad = edad.loc[edad['age'] == int(age), 'fraudes_por_edad'].values[0]

    hora = st.time_input("Hora de la transacción")
    hour = hora.hour
    h = pd.read_csv("hora.csv")
    fraudes_por_hora = h.loc[h['hour'] == int(hour), 'fraudes_por_hora'].values[0]

    dia = trans_date.day
    day = pd.read_csv("dia.csv")
    fraudes_por_dia = day.loc[day['day_of_month'] == int(dia), 'fraudes_por_día'].values[0]

    data = {
        "amt": amt,
        "city_pop": city_pop,
        "distancia": distancia,
        "fraudes_por_Categoria": fraudes_por_Categoria,
        "fraudes_por_estado": fraudes_por_estado,
        "fraudes_por_edad": fraudes_por_edad,
        "fraudes_por_hora": fraudes_por_hora,
        "fraudes_por_día": fraudes_por_dia
    }
 
    datos_de_entrada = pd.DataFrame(data, index=[0])

    # Realizar la predicción utilizando el modelo cargado
    resultado_prediccion = modelo_cargado.predict(datos_de_entrada)

    # Convertir el resultado a una etiqueta legible para el usuario
    etiqueta_prediccion = "Transacción fraudulenta" if resultado_prediccion == 1 else "Transacción legítima"

    # Mostrar el resultado de la predicción al usuario
    st.write('El resultado de la predicción es:', etiqueta_prediccion)

    st.markdown("Puedes consultar en la siguiente tabla la población de cada ciudad:")
    df = pd.read_csv("city_pop.csv")
    st.dataframe(df)


if menu == 'Todos los modelos de predicción':
    # Obtener la ruta absoluta de la carpeta padre
    parent_folder = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    # Mapeo de nombres de modelo a tipos de modelo
    model_types = {
        "Model1": "RandomForestClassifier",
        "Model2": "GradientBoostingClassifier",
        "Model3": "KNeighborsClassifier",
        "Model4": "XGBClassifier",
        "Model5": "LogisticRegressionClassifier",
        "Model6": "SVC",
        "Model7": "RandomForestClassifier con PCA",
        "Model8": "Sequential de Keras"
    }

    # Obtener la lista de subcarpetas de la carpeta "models"
    model_folders = [folder for folder in os.listdir(os.path.join(parent_folder, "models")) if os.path.isdir(os.path.join(parent_folder, "models", folder))]

    # Crear una lista de nombres de tipos de modelo
    model_types_list = [model_types[folder] for folder in model_folders]
    
    # Selector de tipo de modelo
    selected_model_type = st.selectbox("Seleccione el tipo de modelo:", model_types_list)

    # Obtener el nombre de la carpeta del modelo seleccionado
    selected_model_folder = [folder for folder, model_type in model_types.items() if model_type == selected_model_type][0]

    # Obtener la ruta del modelo seleccionado
    model_path = os.path.join(parent_folder, "models", selected_model_folder, "trained_model.pkl" if selected_model_folder != "Model8" else "trained_model.h5")

    # Cargar el modelo correspondiente
    if selected_model_folder != "Model8":
        modelo_cargado = joblib.load(model_path)
    else:
        modelo_cargado = load_model(model_path)
    # Recoger las características de entrada del usuario
    amt = st.slider("Cuantía de la transacción", 1, 30000, 50)
    city_pop = st.slider("Población de la ciudad", 20, 3000000)
    lat = st.slider("Latitud del domicilio cliente", 19.02779, 67.51027, 0.00001)
    long = st.slider("Longitud del domicilio cliente", -166.6712, -66.9509, 0.00001)
    lat_merch = st.slider("Latitud del comercio", 19.02779, 67.51027, 0.00001)
    long_merch = st.slider("Longitud del comercio", -166.6712, -66.9509, 0.00001)
    distancia = calcular_distancia(lat, long, lat_merch, long_merch)

    categoria_options = ('misc_net', 'grocery_pos', 'entertainment', 'gas_transport',
                         'misc_pos', 'grocery_net', 'shopping_net', 'shopping_pos',
                         'food_dining', 'personal_care', 'health_fitness', 'travel',
                         'kids_pets', 'home')
    categoria = st.selectbox("Seleccione la categoría del comercio:", categoria_options)

    cat = pd.read_csv("categoria.csv")
    fraudes_por_Categoria = cat.loc[cat['category'] == categoria, 'fraudes_por_Categoria'].iloc[0]

    state_options = ('NC', 'WA', 'ID', 'MT', 'VA', 'PA', 'KS', 'TN', 'IA', 'WV', 'FL',
                     'CA', 'NM', 'NJ', 'OK', 'IN', 'MA', 'TX', 'WI', 'MI', 'WY', 'HI',
                     'NE', 'OR', 'LA', 'DC', 'KY', 'NY', 'MS', 'UT', 'AL', 'AR', 'MD',
                     'GA', 'ME', 'AZ', 'MN', 'OH', 'CO', 'VT', 'MO', 'SC', 'NV', 'IL',
                     'NH', 'SD', 'AK', 'ND', 'CT', 'RI', 'DE')
    estado = st.selectbox("Seleccione el estado:", state_options)

    state = pd.read_csv("estado.csv")
    fraudes_por_estado = state.loc[state['state'] == estado, 'fraudes_por_estado'].iloc[0]

    # Obtener la fecha y hora de transacción bancaria y la fecha de nacimiento
    trans_date = st.date_input("Fecha Transacción Bancaria")
    # Set the default start date to "01/01/1920"
    default_start_date = date(1920, 1, 1)

    # Create the date input widget
    dob = st.date_input("Fecha de Nacimiento", min_value=default_start_date)
    age = relativedelta(trans_date, dob).years
    
    edad = pd.read_csv("edad.csv")
    fraudes_por_edad = 0  # Valor predeterminado si no se encuentra una coincidencia

    if int(age) in edad['age'].values:
        fraudes_por_edad = edad.loc[edad['age'] == int(age), 'fraudes_por_edad'].values[0]

    hora = st.time_input("Hora de la transacción")
    hour = hora.hour
    h = pd.read_csv("hora.csv")
    fraudes_por_hora = h.loc[h['hour'] == int(hour), 'fraudes_por_hora'].values[0]

    dia = trans_date.day
    day = pd.read_csv("dia.csv")
    fraudes_por_dia = day.loc[day['day_of_month'] == int(dia), 'fraudes_por_día'].values[0]

    data = {
        "amt": amt,
        "city_pop": city_pop,
        "distancia": distancia,
        "fraudes_por_Categoria": fraudes_por_Categoria,
        "fraudes_por_estado": fraudes_por_estado,
        "fraudes_por_edad": fraudes_por_edad,
        "fraudes_por_hora": fraudes_por_hora,
        "fraudes_por_día": fraudes_por_dia
    }
 
    datos_de_entrada = pd.DataFrame(data, index=[0])

    # Realizar la predicción utilizando el modelo cargado
    resultado_prediccion = modelo_cargado.predict(datos_de_entrada)

    # Redondear la predicción si es un modelo Sequential de Keras (Model8)
    if selected_model_folder == "Model8":
        resultado_prediccion = np.round(resultado_prediccion)

    # Convertir el resultado a una etiqueta legible para el usuario
    etiqueta_prediccion = "Transacción fraudulenta" if resultado_prediccion == 1 else "Transacción legítima"

    # Mostrar el resultado de la predicción al usuario
    st.write('El resultado de la predicción es:', etiqueta_prediccion)

    st.markdown("Puedes consultar en la siguiente tabla la población de cada ciudad:")
    df = pd.read_csv("city_pop.csv")
    st.dataframe(df)

if menu == 'Evaluación de modelos de predicción':
    # Obtener la ruta absoluta de la carpeta padre
    parent_folder = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    # Mapeo de nombres de modelo a tipos de modelo
    model_types = {
        "Model1": "RandomForestClassifier",
        "Model2": "GradientBoostingClassifier",
        "Model3": "KNeighborsClassifier",
        "Model4": "XGBClassifier",
        "Model5": "LogisticRegressionClassifier",
        "Model6": "SVC",
        "Model7": "RandomForestClassifier con PCA",
        "Model8": "Sequential de Keras"
    }

    # Obtener la lista de subcarpetas de la carpeta "models"
    model_folders = [folder for folder in os.listdir(os.path.join(parent_folder, "models")) if os.path.isdir(os.path.join(parent_folder, "models", folder))]

    # Crear una lista de nombres de tipos de modelo
    model_types_list = [model_types[folder] for folder in model_folders]

    # Selector de tipo de modelo
    selected_model_type = st.selectbox("Seleccione el tipo de modelo:", model_types_list)

    # Obtener el nombre de la carpeta del modelo seleccionado
    selected_model_folder = [folder for folder, model_type in model_types.items() if model_type == selected_model_type][0]

    # Obtener la ruta del modelo seleccionado
    model_path = os.path.join(parent_folder, "models", selected_model_folder, "trained_model.pkl" if selected_model_folder != "Model8" else "trained_model.h5")

    # Cargar el modelo correspondiente
    if selected_model_folder != "Model8":
        modelo_cargado = joblib.load(model_path)
    else:
        modelo_cargado = load_model(model_path)

    # Leer el archivo CSV de prueba
    df = pd.read_csv("../data/test/test.csv")

    # Obtener las características de prueba (X_test) y las etiquetas de prueba (y_test)
    X_test = df[['amt', 'city_pop', 'distancia', 'fraudes_por_Categoria',
        'fraudes_por_estado', 'fraudes_por_edad', 'fraudes_por_hora',
        'fraudes_por_día']]
    y_test = df['is_fraud']

    # Realizar la predicción utilizando el modelo cargado
    predictions = modelo_cargado.predict(X_test)

    # Redondear la predicción si es un modelo Sequential de Keras (Model8)
    if selected_model_folder == "Model8":
        predictions = np.round(predictions)

        # Calcular la matriz de confusión
    c_matrix = confusion_matrix(y_test, predictions, normalize='true')
    st.write("Matriz de confusión:")
    st.dataframe(pd.DataFrame(c_matrix))

    # Mostrar la matriz de confusión como un mapa de calor
    plt.figure(figsize=(10, 10))
    sns.heatmap(c_matrix, annot=True)
    plt.title("Matriz de confusión para " + selected_model_folder)
    st.pyplot(plt)

    # Calcular y mostrar las métricas de evaluación
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, predictions)

    st.write("Accuracy score:", accuracy)
    st.write("Precision score:", precision)
    st.write("Recall score:", recall)
    st.write("ROC AUC score:", roc_auc)