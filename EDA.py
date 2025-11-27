import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""
[0 , 1] es Rango - [0 - 1] es BInario

danceability      [0 , 1]
energy            [0 , 1]
key               [0 - 11]
loudness          [-49.253 , 3.744]
mode              [0 - 1]
speechiness       [0 , 1]
acousticness      [0 , 1]
instrumentalness  [0 , 1]
liveness          [0 , 1]
valence           [0 , 1]
tempo             [0 , 241.423]
duration_ms       [15168 , 4170227]
time_signature    [0 , 5]
chorus_hit        [0 , 433.182]
sections          [0 , 169]
target            [0 - 1]
"""

def calcularCorrelacion(df,var1,var2):
    correlacion = df[var1].corr(df[var2])
    print(f"Correlación entre {var1} y {var2}: {correlacion:.3f}")

    # Interpretación de la correlación
    if abs(correlacion) > 0.7:
        strength = "fuerte"
    elif abs(correlacion) > 0.3:
        strength = "moderada"
    else:
        strength = "débil"

    direction = "positiva" if correlacion > 0 else "negativa"
    print(f"Relación {strength} y {direction}")
    return correlacion

def graficoDispersion(df, var1, var2, correlacion):
    plt.figure(figsize=(10, 6))
    
    # Gráfico de dispersión
    plt.scatter(df[var1], df[var2], alpha=0.5, color='royalblue')
    
    # Línea de tendencia
    z = np.polyfit(df[var1], df[var2], 1)
    p = np.poly1d(z)
    plt.plot(df[var1], p(df[var1]), color='red', linewidth=2, label='Línea de tendencia')
    
    plt.title(f'Relación entre {var1.title()} y {var2.title()}\nCorrelación: {correlacion:.3f}', fontsize=14)
    plt.xlabel(var1.title(), fontsize=12)
    plt.ylabel(var2.title(), fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

#========================================================================
# MAIN
#========================================================================

#====================================
# 1 - CARGA DE DATOS
#====================================

ruta = r"C:\Users\bruni\Desktop\Coding\Python\proj-sem-analisis-de-datos\data"

# Listar todos los archivos CSV de la carpeta
archivos = [os.path.join(ruta, f) for f in os.listdir(ruta) if f.endswith(".csv")]

# Leer y concatenar todos los CSV
dfs = []
for archivo in archivos:
    df_temp = pd.read_csv(archivo)
    nombre_archivo = os.path.basename(archivo).replace(".csv", "")
    df_temp["decada"] = nombre_archivo.split('-')[-1].replace('s','')
    dfs.append(df_temp)

df = pd.concat(dfs, ignore_index=True)

#print(df.shape)  # dimensiones
print(df.head())  # primeras filas

#====================================
# 2 - RESUMEN ESTADISTICO
#====================================
# resumen rapido
desc_res=df.describe().T
print(desc_res)
"""
# resumen mas controlado
res_estad = pd.DataFrame({
    'Media': df.mean(numeric_only=True),
    'Mediana': df.median(numeric_only=True),
    'Desv.Estándar': df.std(numeric_only=True),
    'Mínimo': df.min(numeric_only=True),
    'Máximo': df.max(numeric_only=True)
})

#print(desc_res)
#print("\n\n")
"""

# Resumen de columnas en especifico
"""
cols = ['danceability','energy','tempo']
res_estad_col = pd.DataFrame({
    'Media': df[cols].mean(),
    'Mediana': df[cols].median(),
    'Desv.Estándar': df[cols].std(),
    'Mínimo': df[cols].min(),
    'Máximo': df[cols].max()
})
"""

#====================================
# 3 HISTOGRAMA Y BOXPLOT
#====================================

#=======HISTOGRAMA=======
# Histograma de una variable en especifico
"""
plt.hist(df['danceability'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribución de Danceability')
plt.xlabel('Danceability')
plt.ylabel('Frecuencia')
plt.show()"""

# Histograma de todads las variables numericas
df.hist(bins=30, figsize=(15, 10), color='skyblue', edgecolor='black')
plt.suptitle('Distribución de variables numéricas', fontsize=16)
plt.show

#=======BOXPLOT=======
# Boxplot de variables en especifico

df[['danceability','energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']].boxplot(figsize=(10,6))
plt.title('Boxplots de variables seleccionadas')
plt.show()

# Boxplot de todads las variables numericas (Recomendable normalizar antes)
"""
df.select_dtypes(include='number').plot(kind='box', figsize=(15,8), rot=45, title='Boxplots de todas las variables numéricas')
plt.show()"""

#====================================
# 4 - CORRELACION
#====================================
# Seleccionar dos variables de interés
var1 = 'energy'
var2 = 'loudness'

correlacion = calcularCorrelacion(df, var1, var2)
graficoDispersion(df, var1, var2, correlacion)

#====================================
# 5 - ANALISIS DE FRECUENCIA
#====================================

print("\nColumnas disponibles:")
print(df.columns.tolist())

columna = "danceability"
inter = 10 #cantidad de intervalos

plt.figure(figsize=(10, 5))
plt.hist(df[columna], bins=inter, color='lightblue', edgecolor='black', alpha=0.7)
plt.title(f'Histograma de {columna}')
plt.xlabel(columna)
plt.ylabel('Frecuencia')
plt.grid(True, alpha=0.3)
plt.show()

min_val = df[columna].min()
max_val = df[columna].max()
intervalos = np.linspace(min_val, max_val, inter + 1) #divide el df en intervalos iguales

frecuencias = []
for i in range(len(intervalos)-1):
    inicio = intervalos[i]
    fin = intervalos[i+1]
    count = len(df[(df[columna] >= inicio) & (df[columna] < fin)])
    frecuencias.append([f"{inicio:.2f}-{fin:.2f}", count])

# Último intervalo incluye el máximo
frecuencias[-1][0] = f"{intervalos[-2]:.2f}-{intervalos[-1]:.2f}"

df_frecuencias = pd.DataFrame(frecuencias, columns=['Intervalo', 'Frecuencia'])
print("Tabla de frecuencias:")
print(df_frecuencias)


"""Discretización simple en categorías"""
print(f"\nDiscretización de {columna} en {inter} categorías:")

# Crear categorías
categorias = pd.cut(df[columna], bins=inter, labels=[f'Cat_{i+1}' for i in range(inter)])
df[f'{columna}_categoria'] = categorias

print(df[f'{columna}_categoria'].value_counts().sort_index())


"""Normalización simple Min-Max"""
print(f"\nNormalización de {columna}:")

min_val = df[columna].min()
max_val = df[columna].max()

df[f'{columna}_normalizado'] = (df[columna] - min_val) / (max_val - min_val)

print(f"Original: {min_val:.2f} a {max_val:.2f}")
print(f"Normalizado: {df[f'{columna}_normalizado'].min():.2f} a {df[f'{columna}_normalizado'].max():.2f}")


print("\n" + "="*50 + "\n")