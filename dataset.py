import pandas as pd
import numpy as np
import random

# -------------------------------
# 1. Definición de parámetros
# -------------------------------
num_instancias = 1100  # cantidad de correos a generar
np.random.seed(42)     # semilla para reproducibilidad

# -------------------------------
# 2. Función para generar una fila
# -------------------------------
def generar_correo():
    # Clase objetivo: 0 = HAM, 1 = SPAM
    clase = np.random.choice([0, 1], p=[0.5, 0.5])  # mitad HAM, mitad SPAM
    
    # Features
    if clase == 1:  # SPAM
        longitud_mensaje = np.random.randint(200, 2000)
        num_palabras_mayuscula = np.random.randint(3, 15)
        num_signos_exclamacion = np.random.randint(1, 10)
        num_links = np.random.randint(1, 5)
        num_palabras_dinero = np.random.randint(1, 6)
        contiene_adjuntos = np.random.choice([0, 1], p=[0.6, 0.4])
        porcentaje_numeros = round(np.random.uniform(0.05, 0.3), 2)
        es_respuesta = 0
        num_palabras_raras = np.random.randint(5, 20)
        
        # Generamos hora con distribución sesgada hacia horarios extraños
        probs = [0.02]*6 + [0.05]*6 + [0.1]*6 + [0.03]*6
        probs = np.array(probs) / sum(probs)  # Normalizamos
        hora_envio = np.random.choice(range(24), p=probs)
    
    else:  # HAM
        longitud_mensaje = np.random.randint(50, 1500)
        num_palabras_mayuscula = np.random.randint(0, 5)
        num_signos_exclamacion = np.random.randint(0, 3)
        num_links = np.random.randint(0, 2)
        num_palabras_dinero = np.random.randint(0, 2)
        contiene_adjuntos = np.random.choice([0, 1], p=[0.8, 0.2])
        porcentaje_numeros = round(np.random.uniform(0.0, 0.1), 2)
        es_respuesta = np.random.choice([0, 1], p=[0.3, 0.7])
        num_palabras_raras = np.random.randint(0, 5)
        hora_envio = np.random.randint(7, 20)  # horas de oficina
    
    return [
        longitud_mensaje,
        num_palabras_mayuscula,
        num_signos_exclamacion,
        num_links,
        num_palabras_dinero,
        contiene_adjuntos,
        porcentaje_numeros,
        es_respuesta,
        num_palabras_raras,
        hora_envio,
        clase
    ]

# -------------------------------
# 3. Generación del dataset
# -------------------------------
datos = [generar_correo() for _ in range(num_instancias)]

columnas = [
    "longitud_mensaje",
    "num_palabras_mayuscula",
    "num_signos_exclamacion",
    "num_links",
    "num_palabras_dinero",
    "contiene_adjuntos",
    "porcentaje_numeros",
    "es_respuesta",
    "num_palabras_raras",
    "hora_envio",
    "clase"  # 0 = HAM, 1 = SPAM
]

df = pd.DataFrame(datos, columns=columnas)

# -------------------------------
# 4. Guardar en CSV
# -------------------------------
df.to_csv("dataset_spam_ham.csv", index=False)

print("✅ Dataset generado con éxito: dataset_spam_ham.csv")
print(df.head())
