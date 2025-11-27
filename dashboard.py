import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------------------------------------------------------
# 1. Cargar todos los datasets
# ---------------------------------------------------------
st.title("Dashboard Interactivo – Análisis Spotify Hit Predictor Dataset")

# Definir la ruta de la carpeta data
data_folder = 'C:/Users/bruni/Desktop/Coding/Python/proj-sem-analisis-de-datos/data/'

# Lista de archivos en la carpeta data (solo CSVs)
archivos = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

# Crear diccionario con nombres más amigables para mostrar
nombres_datasets = {
    'dataset-of-60s.csv': '1960s',
    'dataset-of-70s.csv': '1970s',
    'dataset-of-80s.csv': '1980s',
    'dataset-of-90s.csv': '1990s',
    'dataset-of-00s.csv': '2000s',
    'dataset-of-10s.csv': '2010s'
}

# SelectBox para elegir dataset
dataset_seleccionado = st.sidebar.selectbox(
    "Selecciona el dataset:",
    options=list(nombres_datasets.keys()),
    format_func=lambda x: nombres_datasets[x]
)

# Cargar el dataset seleccionado
df = pd.read_csv(os.path.join(data_folder, dataset_seleccionado))

# Mostrar información del dataset actual
st.sidebar.info(f"Filas: {df.shape[0]} | Columnas: {df.shape[1]}")

# Botón para resetear filtros
if st.sidebar.button("Resetear todos los filtros"):
    st.rerun()

st.subheader(f"Vista general del dataset: {nombres_datasets[dataset_seleccionado]}")

st.write("Primeras 100 filas del dataset:")
st.dataframe(df.head(100))

st.write(f"Filas: {df.shape[0]}  |  Columnas: {df.shape[1]}")

# Información general
st.write("Descripción estadística:")
st.dataframe(df.describe())

# ---------------------------------------------------------
# 2. Filtros interactivos para TODAS las columnas numéricas
# ---------------------------------------------------------
st.sidebar.header("Filtros")

# Identificar columnas numéricas
numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Separar en variables continuas y discretas
continuous_columns = []
discrete_columns = []

for col in numeric_columns:
    # Considerar como discreta si tiene menos de 20 valores únicos, de lo contrario continua
    if df[col].nunique() <= 20:
        discrete_columns.append(col)
    else:
        continuous_columns.append(col)

# Aplicar filtros
df_filtrado = df.copy()

# Filtros para variables continuas (sliders)
if continuous_columns:
    st.sidebar.subheader("Variables Continuas")
    for col in continuous_columns:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        current_min, current_max = st.sidebar.slider(
            f"{col}",
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val),
            key=f"cont_{col}"
        )
        df_filtrado = df_filtrado[(df_filtrado[col] >= current_min) & (df_filtrado[col] <= current_max)]

# Filtros para variables discretas (selectbox o multiselect)
if discrete_columns:
    st.sidebar.subheader("Variables Discretas")
    for col in discrete_columns:
        unique_vals = sorted(df[col].unique())
        # Si hay muchos valores únicos, usar slider, si no multiselect
        if len(unique_vals) > 12:
            min_val = int(df[col].min())
            max_val = int(df[col].max())
            selected_min, selected_max = st.sidebar.slider(
                f"{col}",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val),
                key=f"disc_slider_{col}"
            )
            df_filtrado = df_filtrado[(df_filtrado[col] >= selected_min) & (df_filtrado[col] <= selected_max)]
        else:
            selected_vals = st.sidebar.multiselect(
                f"{col}",
                options=unique_vals,
                default=unique_vals,
                key=f"disc_multi_{col}"
            )
            if selected_vals:
                df_filtrado = df_filtrado[df_filtrado[col].isin(selected_vals)]

# Mostrar resumen de filtros aplicados
st.sidebar.header("Resumen de Filtros")
st.sidebar.write(f"Filas originales: {df.shape[0]}")
st.sidebar.write(f"Filas filtradas: {df_filtrado.shape[0]}")
st.sidebar.write(f"Porcentaje conservado: {((df_filtrado.shape[0] / df.shape[0]) * 100):.1f}%")

# ---------------------------------------------------------
# 2.1. Mostrar datos filtrados
# ---------------------------------------------------------
st.subheader("Datos filtrados")
st.dataframe(df_filtrado)
st.write(f"Filas: {df_filtrado.shape[0]}  |  Columnas: {df_filtrado.shape[1]}")

# Descripción de datos filtrados
st.write("Descripción estadística (Filtrados):")
st.dataframe(df_filtrado.describe())

# ---------------------------------------------------------
# 3. Histogramas interactivos
# ---------------------------------------------------------
st.subheader("Histograma")

col_hist = st.selectbox("Selecciona columna numérica:", numeric_columns)

fig_hist = px.histogram(df_filtrado, x=col_hist, nbins=30, 
                       title=f"Histograma de {col_hist} - {nombres_datasets[dataset_seleccionado]}")
st.plotly_chart(fig_hist)

# ---------------------------------------------------------
# 4. Boxplot por Hit/No-Hit
# ---------------------------------------------------------
st.subheader("Boxplot por Hit/No-Hit")

# Verificar si existe la columna 'target' que indica hit (1) o no-hit (0)
if 'target' in df_filtrado.columns:
    col_box = st.selectbox("Selecciona columna para boxplot:", numeric_columns)
    
    # Crear boxplot comparando hit vs no-hit
    fig_box = px.box(df_filtrado, 
                     x='target', 
                     y=col_box, 
                     color='target',
                     title=f"Distribución de {col_box} por Hit/No-Hit - {nombres_datasets[dataset_seleccionado]}",
                     labels={'target': 'Tipo de Canción', col_box: col_box},
                     category_orders={'target': [0, 1]})
    
    # Personalizar los nombres de las categorías
    fig_box.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=[0, 1],
            ticktext=['No-Hit', 'Hit']
        )
    )
    
    # Personalizar colores
    fig_box.update_traces(marker_color='lightcoral', 
                         selector=dict(name='0'))
    fig_box.update_traces(marker_color='lightgreen', 
                         selector=dict(name='1'))
    
    st.plotly_chart(fig_box)
    
    # Agregar estadísticas descriptivas
    st.write(f"**Estadísticas descriptivas para {col_box}:**")
    
    # Calcular estadísticas por grupo
    stats_hit = df_filtrado[df_filtrado['target'] == 1][col_box].describe()
    stats_nohit = df_filtrado[df_filtrado['target'] == 0][col_box].describe()
    
    # Crear DataFrame comparativo
    stats_comparativo = pd.DataFrame({
        'Hit': stats_hit,
        'No-Hit': stats_nohit
    })
    
    st.dataframe(stats_comparativo)
    
else:
    st.warning("No se encontró la columna 'target' en el dataset. No se puede generar el boxplot comparativo.")
    
    # Mostrar boxplot normal como fallback
    col_box = st.selectbox("Selecciona columna para boxplot:", numeric_columns)
    fig_box = px.box(df_filtrado, y=col_box, title=f"Boxplot de {col_box} - {nombres_datasets[dataset_seleccionado]}")
    st.plotly_chart(fig_box)

# ---------------------------------------------------------
# 5. Mapa de calor de correlación
# ---------------------------------------------------------
st.subheader("Mapa de correlación")

corr = df_filtrado.select_dtypes(include=["int64", "float64"]).corr()

fig_corr, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="viridis", ax=ax)
plt.title(f"Mapa de Correlación - {nombres_datasets[dataset_seleccionado]}")
st.pyplot(fig_corr)

#-----------------------------------------------------------------
# Mapa de correlacion filtrado
#-----------------------------------------------------------------
st.subheader("Mapa de correlación (Filtrado)")

# Widget para seleccionar columnas
selected_columns = st.multiselect("Selecciona las columnas a mostrar", numeric_columns, 
                                 default=numeric_columns, key="corr_columns")

if selected_columns:  # solo si hay columnas seleccionadas
    # Calcula la correlación solo de las columnas seleccionadas
    corr = df_filtrado[selected_columns].corr()

    # Crear la figura del heatmap
    fig_corr, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="viridis", ax=ax)
    plt.title(f"Mapa de Correlación (Filtrado) - {nombres_datasets[dataset_seleccionado]}")
    st.pyplot(fig_corr)
else:
    st.warning("Por favor selecciona al menos una columna para mostrar la correlación.")

# ---------------------------------------------------------
# 6. Gráfico de dispersión
# ---------------------------------------------------------
st.subheader("Gráfico de dispersión")

x_col = st.selectbox("Eje X:", numeric_columns, index=0, key="scatter_x")
y_col = st.selectbox("Eje Y:", numeric_columns, index=1, key="scatter_y")

fig_scatter = px.scatter(df_filtrado, x=x_col, y=y_col, trendline="ols",
                         title=f"Relación entre {x_col} y {y_col} - {nombres_datasets[dataset_seleccionado]}")

fig_scatter.update_traces(line=dict(color='red'), selector=dict(mode='lines'))

st.plotly_chart(fig_scatter)

# ---------------------------------------------------------
# 7. Descargar datos filtrados
# ---------------------------------------------------------
st.subheader("Descargar datos filtrados")

csv = df_filtrado.to_csv(index=False).encode('utf-8')
st.download_button(
    label=f"Descargar {nombres_datasets[dataset_seleccionado]} CSV filtrado",
    data=csv,
    file_name=f"datos_filtrados_{nombres_datasets[dataset_seleccionado]}.csv",
    mime="text/csv"
)
