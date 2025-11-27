import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Configurar estilo de gráficos
plt.style.use('default')
sns.set_palette("husl")

# 1. PREPARAR DATOS
def preparar_datos(df):
    # Seleccionar features (excluir metadata que no son características musicales)
    features = ['danceability', 'energy', 'loudness', 'speechiness', 
                'acousticness', 'instrumentalness', 'liveness', 'valence', 
                'tempo', 'duration_ms', 'key', 'mode', 'time_signature']
    
    X = df[features]
    y = df['target']  # 0 = no hit, 1 = hit
    
    return X, y

# Cargar datos
data = pd.read_csv('C:/Users/bruni/Desktop/Coding/Python/proj-sem-analisis-de-datos/data/dataset-of-10s.csv')

X, y = preparar_datos(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Preanálisis de métodos

# 2.1 Escalar datos (importante para SVM y MLP)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2.2 Modelos a comparar
models = {
    'SVM': SVC(kernel='rbf', random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# 2.3 Entrenar y evaluar modelos
results = {}
for name, model in models.items():
    if name in ['SVM']:
        # Usar datos escalados
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        # Usar datos originales
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"{name}: {accuracy:.4f}")


print("\n📊 DATASET PARA MODELADO:")
print(f"Features: {X.shape[1]} variables")
print(f"Total muestras: {X.shape[0]}")
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
print(f"Distribución target - Train: {np.bincount(y_train)}")
print(f"Distribución target - Test: {np.bincount(y_test)}")

# 3. ENTRENAR RANDOM FOREST
print("\n🎯 ENTRENANDO RANDOM FOREST...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1  # Usar todos los cores del CPU
)

rf_model.fit(X_train, y_train)

# 4. PREDICCIONES Y MÉTRICAS
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]  # Probabilidades para la clase 1 (hit)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ ACCURACY: {accuracy:.4f}")

# 5. MATRIZ DE CONFUSIÓN DETALLADA
print("\n🔍 MATRIZ DE CONFUSIÓN DETALLADA:")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 4))

# Subplot 1: Matriz de confusión numérica
plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Hit (0)', 'Hit (1)'], 
            yticklabels=['No Hit (0)', 'Hit (1)'])
plt.title('Matriz de Confusión\nRandom Forest')
plt.ylabel('Real')
plt.xlabel('Predicho')

# Subplot 2: Matriz de confusión normalizada
plt.subplot(1, 2, 2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=['No Hit (0)', 'Hit (1)'], 
            yticklabels=['No Hit (0)', 'Hit (1)'])
plt.title('Matriz de Confusión (%)\nRandom Forest')
plt.ylabel('Real')
plt.xlabel('Predicho')

plt.tight_layout()
plt.show()

# 6. MÉTRICAS DETALLADAS
print("\n📈 REPORTE DE CLASIFICACIÓN:")
print(classification_report(y_test, y_pred, target_names=['No Hit', 'Hit']))

# Calcular métricas manualmente para más control
tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n📊 MÉTRICAS CALCULADAS:")
print(f"Precision: {precision:.4f}  (De los predichos como Hit, cuántos son realmente Hit)")
print(f"Recall:    {recall:.4f}    (De los Hits reales, cuántos logramos identificar)")
print(f"F1-Score:  {f1:.4f}     (Balance entre Precision y Recall)")
print(f"Accuracy:  {accuracy:.4f}  (Total de predicciones correctas)")

# 7. IMPORTANCIA DE VARIABLES
print("\n🎵 ANÁLISIS DE IMPORTANCIA DE VARIABLES:")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTOP 10 VARIABLES MÁS IMPORTANTES:")
for i, row in feature_importance.head(10).iterrows():
    print(f"{row['feature']:<20}: {row['importance']:.4f} ({row['importance']*100:.2f}%)")

# 8. VISUALIZACIÓN DE IMPORTANCIA
plt.figure(figsize=(10, 8))

# Gráfico de barras horizontal
plt.subplot(2, 1, 1)
colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importance)))
bars = plt.barh(feature_importance['feature'], feature_importance['importance'], color=colors)
plt.xlabel('Importancia')
plt.title('Importancia de Variables 2010 - Random Forest\n(Para predecir Hit vs No Hit)')
plt.gca().invert_yaxis()

# Añadir valores en las barras
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{width:.3f}', ha='left', va='center', fontsize=9)

# Gráfico de pie para las top 5
plt.subplot(2, 2, 3)
top_5 = feature_importance.head(5)
plt.pie(top_5['importance'], labels=top_5['feature'], autopct='%1.1f%%', startangle=90)
plt.title('Top 5 Variables Más Importantes')

# Gráfico de acumulado
plt.subplot(2, 2, 4)
cumulative_importance = feature_importance['importance'].cumsum()
plt.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 'o-', linewidth=2)
plt.xlabel('Número de Variables')
plt.ylabel('Importancia Acumulada')
plt.title('Importancia Acumulada de Variables')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n💡 INTERPRETACIÓN MUSICAL:")
top_3 = feature_importance.head(3)['feature'].tolist()
print(f"Las 3 características más importantes para predecir un HIT son:")
for i, feature in enumerate(top_3, 1):
    print(f"  {i}. {feature}")
