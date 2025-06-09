# src/predict_console.py
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Cargar modelos
modelos = {
    "Logistic Regression": joblib.load("models/LogisticRegression.pkl"),
    "Random Forest": joblib.load("models/RandomForest.pkl"),
    "KNN": joblib.load("models/K-NearestNeighbors.pkl"),
    "SVM": joblib.load("models/SupportVectorMachine.pkl")
}

# Cargar escalador
scaler = joblib.load("models/scaler.pkl")

# Solicitar valores al usuario
def pedir_valores():
    print("🔎 Ingrese los datos del paciente:")
    edad_paciente = float(input("Edad del paciente (años): "))
    edad_gestacion = float(input("Edad de la gestación (semanas): "))
    proteina_urinaria = float(input("Nivel de proteína urinaria (mg/dl): "))
    antecedentes_preeclampsia = int(input("¿Antecedentes de preeclampsia? (1: Sí, 0: No): "))
    glucosa = float(input("Nivel de glucosa (mg/dl): "))
    plaquetas = float(input("Recuento de plaquetas (/µL): "))
    arteria_uterina = float(input("Índice de resistencia de arteria uterina: "))

    return np.array([[edad_paciente, edad_gestacion, proteina_urinaria,
                      antecedentes_preeclampsia, glucosa, plaquetas,
                      arteria_uterina]])

# Predicción y visualización
def predecir_e_imprimir(X):
    X_scaled = scaler.transform(X)
    resultados = {}

    for nombre, modelo in modelos.items():
        proba = modelo.predict_proba(X_scaled)[0][1] * 100
        resultados[nombre] = round(proba, 2)
        print(f"{nombre}: {proba:.2f}% de riesgo de preeclampsia")

    return resultados

def graficar(resultados):
    nombres = list(resultados.keys())
    valores = list(resultados.values())

    plt.figure(figsize=(8, 5))
    plt.barh(nombres, valores, color='skyblue')
    plt.xlabel("Riesgo de Preeclampsia (%)")
    plt.title("Comparación entre Modelos")
    plt.xlim(0, 100)
    for i, v in enumerate(valores):
        plt.text(v + 1, i, f"{v}%", va='center')
    plt.tight_layout()
    plt.show()

# Ejecutar predicción
if __name__ == "__main__":
    X = pedir_valores()
    resultados = predecir_e_imprimir(X)
    graficar(resultados)
