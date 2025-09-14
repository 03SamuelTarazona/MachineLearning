# ======================================================
# Clasificación del dataset Iris con varios modelos
# ======================================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeClassifier
from sklearn.metrics import classification_report, accuracy_score

# ======================================================
# 1. Cargar dataset Iris
# ======================================================
data = datasets.load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

print("Primeras filas del dataset:")
print(df.head(), "\n")

print("Clases de plantas:", data.target_names, "\n")

# Variables independientes (X) y dependiente (y)
X = df.iloc[:, :-1]
y = df["target"]

# ======================================================
# 2. Dividir dataset en entrenamiento y prueba
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================================================
# 3. Entrenar modelos
# ======================================================

# --- Regresión Lineal (adaptada para clasificación)
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test).round().astype(int)
print("📌 Resultados con Regresión Lineal básica:")
print("Accuracy:", accuracy_score(y_test, y_pred_lin))
print(classification_report(y_test, y_pred_lin, target_names=data.target_names), "\n")

# --- RidgeClassifier
ridge = RidgeClassifier()
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
print("📌 Resultados con RidgeClassifier:")
print("Accuracy:", accuracy_score(y_test, y_pred_ridge))
print(classification_report(y_test, y_pred_ridge, target_names=data.target_names), "\n")

# --- Regresión Logística
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
print("📌 Resultados con Regresión Logística:")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log, target_names=data.target_names), "\n")

# ======================================================
# 4. Importancia de las características
# ======================================================
coef_df = pd.DataFrame(
    log_reg.coef_,
    columns=data.feature_names,
    index=data.target_names
)
print("📌 Importancia de los Features (coeficientes de la Regresión Logística):\n")
print(coef_df, "\n")

# ======================================================
# 5. Visualización de clases
# ======================================================
plt.figure(figsize=(8,6))
scatter = plt.scatter(
    df["petal length (cm)"],
    df["petal width (cm)"],
    c=y,
    cmap="viridis",
    edgecolor="k"
)

# Corregimos la leyenda
handles, _ = scatter.legend_elements()
plt.legend(handles, list(data.target_names), title="Clases")

plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Clasificación Iris - Visualización de Clases")
plt.show()
