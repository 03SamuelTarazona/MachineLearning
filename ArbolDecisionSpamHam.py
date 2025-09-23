import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import f1_score
from scipy import stats

# ============================
# 1. Cargar dataset Spambase
# ============================
data = fetch_openml(data_id=44, as_frame=True)
X, y = data.data, data.target

# Convertir etiquetas a enteros (0 = ham, 1 = spam)
y = y.astype(int)

scores = []      # Accuracy
f1_scores = []   # F1-score

# Crear un solo árbol de decisión
clf = DecisionTreeClassifier(random_state=42)

# ============================
# 2. Repetir evaluación 50 veces
# ============================
for i in range(1, 51):
    # Dividir los datos en entrenamiento y prueba (aleatorio en cada iteración)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True
    )
    
    # Entrenar siempre el mismo árbol
    clf.fit(X_train, y_train)
    
    # Evaluar
    score = clf.score(X_test, y_test)
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    
    # Guardar métricas
    scores.append(score)
    f1_scores.append(f1)

# ============================
# 3. Gráfica evolución Accuracy y F1-score
# ============================
plt.figure(figsize=(8, 5))
plt.plot(scores, marker="o", label="Accuracy")
plt.plot(f1_scores, marker="x", label="F1-Score")
plt.xlabel("Ejecución")
plt.ylabel("Score")
plt.title("Evolución del Accuracy y F1-score en 50 ejecuciones")
plt.legend()
plt.grid(True)
plt.show()

# ============================
# 4. Gráfica evolución Z-score
# ============================
z_scores = stats.zscore(scores)

plt.figure(figsize=(8, 5))
plt.plot(z_scores, marker="s", color="purple", label="Z-score del Accuracy")
plt.axhline(0, color="red", linestyle="--")  # Línea base
plt.xlabel("Ejecución")
plt.ylabel("Z-score")
plt.title("Evolución del Z-score del Accuracy en 50 ejecuciones")
plt.legend()
plt.grid(True)
plt.show()

# ============================
# 5. Visualización resumida del árbol
# ============================
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]  # ordenar de mayor a menor
features = X.columns

# Mostrar en forma de lista Top 10
print("\n📌 Principales preguntas usadas en el árbol (features más importantes):")
for i in range(10):
    print(f"{i+1}. {features[indices[i]]}: {importances[indices[i]]:.4f}")

# Gráfico de pastel con los 5 más importantes
plt.figure(figsize=(6, 6))
plt.pie(importances[indices[:5]], labels=[features[i] for i in indices[:5]], 
        autopct='%1.1f%%', startangle=140)
plt.title("Principales preguntas usadas en el árbol (Top 5)")
plt.show()

# ============================
# 6. Estadísticas finales
# ============================
mean_score = np.mean(scores)
std_score = np.std(scores)

print("Resultados finales después de 50 ejecuciones:")
print(f"Accuracy promedio: {mean_score:.4f}")
print(f"Desviación estándar del accuracy: {std_score:.4f}")
print(f"F1-score promedio: {np.mean(f1_scores):.4f}")
print("Z-scores de los accuracy:", z_scores)
