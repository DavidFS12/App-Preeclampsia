import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib
import os

os.makedirs('models', exist_ok=True)
df = pd.read_csv('data/dataset_simulado.csv')
X = df.drop("riesgo_preeclampsia", axis = 1)
y = df["riesgo_preeclampsia"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, 'models/scaler.pkl')
modelos = {
    "LogisticRegression": LogisticRegression(),
    "RandomForest": RandomForestClassifier(),
    "K-NearestNeighbors": KNeighborsClassifier(),
    "SupportVectorMachine": SVC(probability=True)
}

for nombre, modelo in modelos.items():
    modelo.fit(X_train_scaled, y_train)
    y_pred = modelo.predict(X_test_scaled)
    print(f"\nNombre: {nombre.upper()}:")
    print(classification_report(y_test, y_pred))
    joblib.dump(modelo, f'models/{nombre}.pkl')

print("\nModelos entrenados y guardados en la carpeta models.")
