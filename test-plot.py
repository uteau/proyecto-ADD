import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#========================================================================
# MAIN
#========================================================================

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

print("\nColumnas disponibles:")
print(df.columns.tolist())

columna = "danceability"
bins = 10

plt.figure(figsize=(10, 5))
plt.hist(df[columna], bins=bins, color='lightblue', edgecolor='black', alpha=0.7)
plt.title(f'Histograma de {columna}')
plt.xlabel(columna)
plt.ylabel('Frecuencia')
plt.grid(True, alpha=0.3)
plt.show()

min_val = df[columna].min()
max_val = df[columna].max()
intervalos = np.linspace(min_val, max_val, bins + 1)

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