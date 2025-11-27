import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def cargar_datos(ruta):
    """Carga el dataset desde la ruta"""
    archivos = [os.path.join(ruta, f) for f in os.listdir(ruta) if f.endswith(".csv")]
    dfs = []
    for archivo in archivos:
        df_temp = pd.read_csv(archivo)
        nombre_archivo = os.path.basename(archivo).replace(".csv", "")
        df_temp["decada"] = nombre_archivo.split('-')[-1].replace('s','')
        dfs.append(df_temp)

    df = pd.concat(dfs, ignore_index=True)
    print(f"Dataset cargado: {df.shape}")
    return df

def histograma_y_frecuencias(df, columna, bins=10):
    """Crea histograma y muestra tabla de frecuencias"""
    print(f"\n--- ANÁLISIS DE {columna.upper()} ---")
    
    # Histograma
    plt.figure(figsize=(10, 5))
    plt.hist(df[columna], bins=bins, color='lightblue', edgecolor='black', alpha=0.7)
    plt.title(f'Histograma de {columna}')
    plt.xlabel(columna)
    plt.ylabel('Frecuencia')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Tabla de frecuencias
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

def discretizar_simple(df, columna, bins=5):
    """Discretización simple en categorías"""
    print(f"\nDiscretización de {columna} en {bins} categorías:")
    
    # Crear categorías
    categorias = pd.cut(df[columna], bins=bins, labels=[f'Cat_{i+1}' for i in range(bins)])
    df[f'{columna}_categoria'] = categorias
    
    print(df[f'{columna}_categoria'].value_counts().sort_index())
    return df

def normalizar_simple(df, columna):
    """Normalización simple Min-Max"""
    print(f"\nNormalización de {columna}:")
    
    min_val = df[columna].min()
    max_val = df[columna].max()
    
    df[f'{columna}_normalizado'] = (df[columna] - min_val) / (max_val - min_val)
    
    print(f"Original: {min_val:.2f} a {max_val:.2f}")
    print(f"Normalizado: {df[f'{columna}_normalizado'].min():.2f} a {df[f'{columna}_normalizado'].max():.2f}")
    
    return df

def main():
    # Ruta de los datos
    ruta = r"C:\Users\bruni\Desktop\Coding\Python\proj-sem-analisis-de-datos\data"

    # Cargar datos
    df = cargar_datos(ruta)
    if df is None:
        return
    
    # Mostrar columnas disponibles
    print("\nColumnas disponibles:")
    print(df.columns.tolist())
    
    # Seleccionar algunas variables para análisis
    variables = ['danceability', 'energy', 'loudness', 'tempo']
    
    # Filtrar variables que existen
    variables = [v for v in variables if v in df.columns]
    
    if not variables:
        # Si no existen las variables específicas, usar las primeras numéricas
        variables = df.select_dtypes(include=[np.number]).columns[:3].tolist()
    
    print(f"\nAnalizando variables: {variables}")
    
    # Análisis para cada variable
    for variable in variables[:2]:  # Solo las primeras 2 para simplificar
        try:
            # 1. Histograma y frecuencias
            histograma_y_frecuencias(df, variable)
            
            # 2. Discretización
            df = discretizar_simple(df, variable)
            
            # 3. Normalización
            df = normalizar_simple(df, variable)
            
            print("\n" + "="*50 + "\n")
            
        except Exception as e:
            print(f"Error con {variable}: {e}")
    
    # Mostrar resultados finales
    print("\n--- MUESTRA DE DATOS TRANSFORMADOS ---")
    columnas_transformadas = [col for col in df.columns if '_categoria' in col or '_normalizado' in col]
    if columnas_transformadas:
        print(df[columnas_transformadas].head())
    
    return df

if __name__ == "__main__":
    df_resultado = main()