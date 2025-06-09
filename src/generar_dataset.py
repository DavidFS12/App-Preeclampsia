# src/generate_dataset.py
import pandas as pd
import numpy as np
import random

def generar_dataset(n=500, seed=42):
    np.random.seed(seed)
    random.seed(seed)

    edad_paciente = np.random.randint(14, 45, size=n)
    edad_gestacion = np.random.randint(20, 42, size=n)
    proteina_urinaria = np.random.normal(loc=300, scale=200, size=n).clip(0, 5000)
    antecedentes_preeclampsia = np.random.choice([0, 1], size=n, p=[0.7, 0.3])
    glucosa = np.random.normal(loc=90, scale=25, size=n).clip(40, 250)
    plaquetas = np.random.normal(loc=250000, scale=50000, size=n).clip(50000, 450000)
    arteria_uterina = np.random.normal(loc=1.0, scale=0.3, size=n).clip(0.4, 1.5)

    # Simulamos el target con algo de lógica: mayores niveles aumentan riesgo
    riesgo = (
        (proteina_urinaria > 300).astype(int) +
        (antecedentes_preeclampsia) +
        (glucosa > 140).astype(int) +
        (plaquetas < 100000).astype(int) +
        (arteria_uterina > 1.3).astype(int)
    )
    riesgo_preeclampsia = (riesgo >= 2).astype(int)

    df = pd.DataFrame({
        "edad_paciente": edad_paciente,
        "edad_gestacion": edad_gestacion,
        "proteina_urinaria": proteina_urinaria,
        "antecedentes_preeclampsia": antecedentes_preeclampsia,
        "glucosa": glucosa,
        "plaquetas": plaquetas,
        "arteria_uterina": arteria_uterina,
        "riesgo_preeclampsia": riesgo_preeclampsia
    })

    df.to_csv("data/dataset_simulado.csv", index=False)
    print("✅ Dataset generado y guardado en data/dataset_simulado.csv")

if __name__ == "__main__":
    generar_dataset()
