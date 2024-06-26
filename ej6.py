import numpy as np
from functions import *

def generar_matriz_confusion(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    return TP, FN, TN, FP

def analizar_efectividad(TP, FN, TN, FP):
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + FN + TN + FP)
    print(f"Precisión: {precision:.2f}")
    print(f"Sensibilidad (Recall): {recall:.2f}")
    print(f"Exactitud (Accuracy): {accuracy:.2f}")

# Ejemplo de uso:
# Aquí asumimos que d_test y predicciones son los valores reales y predichos respectivamente.
# Reemplaza estos valores con los resultados de tu modelo.

# Generar matriz de confusión
d_test = cargar_datos("/Users/victoriamarsili/Downloads/chest_xray/test/ALL", escala=128)
predicciones =   predecir(d_test)

TP, FN, TN, FP = generar_matriz_confusion(d_test, predicciones)

# Analizar efectividad
analizar_efectividad(TP, FN, TN, FP)
