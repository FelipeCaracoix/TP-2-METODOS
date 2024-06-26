import json
import numpy as np
import matplotlib.pyplot as plt
from functions import cargar_datos, predecir

def generar_matriz_confusion(y_true, y_pred):
    """
    Genera una matriz de confusión a partir de las etiquetas verdaderas y las predicciones.

    Args:
    - y_true (ndarray): Etiquetas verdaderas.
    - y_pred (ndarray): Predicciones del modelo.

    Returns:
    - tuple: Número de Verdaderos Positivos, Falsos Negativos, Verdaderos Negativos, Falsos Positivos.
    """
    VP = np.sum((y_true == 1) & (y_pred == 1))  # Verdaderos Positivos
    FN = np.sum((y_true == 1) & (y_pred == 0))  # Falsos Negativos
    VN = np.sum((y_true == 0) & (y_pred == 0))  # Verdaderos Negativos
    FP = np.sum((y_true == 0) & (y_pred == 1))  # Falsos Positivos
    return VP, FN, VN, FP

def analizar_efectividad(VP, FN, VN, FP):
    """
    Calcula y muestra las métricas de efectividad: precisión, sensibilidad (recall) y exactitud (accuracy).

    Args:
    - VP (int): Número de Verdaderos Positivos.
    - FN (int): Número de Falsos Negativos.
    - VN (int): Número de Verdaderos Negativos.
    - FP (int): Número de Falsos Positivos.
    """
    precision = VP / (VP + FP)
    recall = VP / (VP + FN)
    accuracy = (VP + VN) / (VP + FN + VN + FP)

    print(f"Precisión: {precision:.2f}")
    print(f"Sensibilidad (Recall): {recall:.2f}")
    print(f"Exactitud (Accuracy): {accuracy:.2f}")

    # Gráfico de barras para visualizar VP, FN, VN, FP
    labels = ['Verdaderos Positivos (VP)', 'Falsos Negativos (FN)', 'Verdaderos Negativos (VN)', 'Falsos Positivos (FP)']
    valores = [VP, FN, VN, FP]
    colores = ['green', 'red', 'blue', 'orange']

    plt.figure(figsize=(8, 6))
    plt.bar(labels, valores, color=colores)
    plt.xlabel('Tipos de Predicciones', fontsize=12)
    plt.ylabel('Cantidad', fontsize=12)
    plt.title('Matriz de Confusión', fontsize=14)
    plt.xticks(rotation=15, fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()

    # Gráfico de pastel para visualizar porcentajes
    porcentajes = [VP / np.sum([VP, FN, VN, FP]), FN / np.sum([VP, FN, VN, FP]),
                   VN / np.sum([VP, FN, VN, FP]), FP / np.sum([VP, FN, VN, FP])]
    etiquetas_pie = [f'{label}\n({porcentaje:.2%})' for label, porcentaje in zip(labels, porcentajes)]
    plt.figure(figsize=(8, 6))
    plt.pie(porcentajes, labels=etiquetas_pie, colors=colores, autopct='%1.1f%%', startangle=140)
    plt.title('Porcentaje de la Matriz de Confusión', fontsize=14)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def main():
    # Cargar datos de prueba
    carpeta_prueba = "/Users/victoriamarsili/Downloads/chest_xray/test/ALL"
    images_test, d_test = cargar_datos(carpeta_prueba, escala=128)

    # Cargar pesos óptimos desde el archivo JSON
    with open('top1.json', 'r') as archivo_json:
        valores_dict = json.load(archivo_json)
        w_estrella = np.array(valores_dict["w_estrella"])
        b_estrella = np.array(valores_dict["b_estrella"])

    # Realizar predicciones
    predicciones = predecir(images_test, w_estrella, b_estrella)

    # Generar matriz de confusión
    VP, FN, VN, FP = generar_matriz_confusion(d_test, predicciones)
    print(len(predicciones))

    # Analizar efectividad y visualizar resultados
    analizar_efectividad(VP, FN, VN, FP)

if __name__ == "__main__":
    main()
