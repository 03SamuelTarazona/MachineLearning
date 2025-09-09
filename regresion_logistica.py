# ======================================================
# Trabajo de Machine Learning
# Clasificación de correos SPAM vs HAM con Regresión Logística
# ======================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, confusion_matrix

# -------------------------------
# 1. Cargar el dataset
# -------------------------------
df = pd.read_csv("dataset_spam_ham.csv")

# Ver primeras filas para inspección
print("Primeras filas del dataset:")
print(df.head())

# -------------------------------
# 2. Separar variables (X) y etiqueta (y)
# -------------------------------
X = df.drop("clase", axis=1)  # Features
y = df["clase"]               # Target (0 = HAM, 1 = SPAM)

# -------------------------------
# 3. Transformación de valores (preprocesamiento)
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 4. Dividir en entrenamiento y prueba
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 5. Entrenar el modelo de Regresión Logística
# -------------------------------
log_reg = LogisticRegression(max_iter=1000, solver="liblinear")
log_reg.fit(X_train, y_train)

# -------------------------------
# 6. Evaluación del modelo
# -------------------------------
y_pred = log_reg.predict(X_test)

# Métrica principal: F1-Score
f1 = f1_score(y_test, y_pred)
print(f"\n✅ F1-Score del modelo: {f1:.3f}")

# Reporte detallado (precisión, recall, f1 por clase)
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=["HAM (0)", "SPAM (1)"]))

# -------------------------------
# 7. Ecuación de la Regresión Logística
# -------------------------------
print("\nCoeficientes del modelo (uno por cada feature):")
for feature, coef in zip(df.drop("clase", axis=1).columns, log_reg.coef_[0]):
    print(f"{feature}: {coef:.4f}")

print(f"\nIntercepto (bias): {log_reg.intercept_[0]:.4f}")

# -------------------------------
# 8. Visualización de resultados (Matriz de confusión)
# -------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["HAM (0)", "SPAM (1)"], yticklabels=["HAM (0)", "SPAM (1)"])
plt.title("Matriz de Confusión - Regresión Logística")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()
