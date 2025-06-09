# app.py
import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Cargar modelos
modelos = {
    "Logistic Regression": joblib.load("models/LogisticRegression.pkl"),
    "Random Forest": joblib.load("models/RandomForest.pkl"),
    "KNN": joblib.load("models/K-NearestNeighbors.pkl"),
    "SVM": joblib.load("models/SupportVectorMachine.pkl")
}
scaler = joblib.load("models/scaler.pkl")

# Título
st.title("🩺 Predicción de Riesgo de Preeclampsia")
st.markdown("Ingrese los valores clínicos del paciente:")

# Formulario
with st.form("formulario_paciente"):
    edad_paciente = st.number_input("Edad del paciente (años)", 15, 50, value=30)
    edad_gestacion = st.number_input("Edad de gestación (semanas)", 20, 42, value=30)
    proteina_urinaria = st.number_input("Proteína urinaria (mg/dl)", 0.0, 500.0, value=100.0)
    antecedentes_preeclampsia = st.selectbox("¿Antecedentes de preeclampsia?", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
    glucosa = st.number_input("Nivel de glucosa (mg/dl)", 50.0, 300.0, value=90.0)
    plaquetas = st.number_input("Recuento de plaquetas (/µL)", 50000, 500000, value=200000)
    arteria_uterina = st.number_input("Índice de arteria uterina", 0.3, 1.2, value=0.6)

    submitted = st.form_submit_button("Predecir")

if submitted:
    X = np.array([[edad_paciente, edad_gestacion, proteina_urinaria,
                   antecedentes_preeclampsia, glucosa, plaquetas,
                   arteria_uterina]])
    X_scaled = scaler.transform(X)

    resultados = {}
    st.subheader("📊 Resultados por Modelo")
    for nombre, modelo in modelos.items():
        prob = modelo.predict_proba(X_scaled)[0][1] * 100
        resultados[nombre] = round(prob, 2)
        st.write(f"**{nombre}**: {resultados[nombre]}% de riesgo")

    # Gráfico
    st.subheader("📈 Comparación de modelos")
    fig, ax = plt.subplots()
    nombres = list(resultados.keys())
    valores = list(resultados.values())

    ax.barh(nombres, valores, color='lightcoral')
    ax.set_xlim([0, 100])
    ax.set_xlabel("Probabilidad de preeclampsia (%)")
    ax.set_title("Comparación de predicciones entre modelos")

    for i, v in enumerate(valores):
        ax.text(v + 1, i, f"{v}%", va='center')
    
    st.pyplot(fig)
