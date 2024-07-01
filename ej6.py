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
    - tuple: Proporciones de Verdaderos Positivos, Falsos Negativos, Verdaderos Negativos, Falsos Positivos.
    """
    VP = np.sum((y_true == 1) & (y_pred == 1))  # Verdaderos Positivos
    FN = np.sum((y_true == 1) & (y_pred == 0))  # Falsos Negativos
    VN = np.sum((y_true == 0) & (y_pred == 0))  # Verdaderos Negativos
    FP = np.sum((y_true == 0) & (y_pred == 1))  # Falsos Positivos
    
    # Calcular proporciones
    total_enfermos = np.sum(y_true == 1)
    total_sanos = np.sum(y_true == 0)
    
    VP_prop = VP / total_enfermos if total_enfermos > 0 else 0
    FN_prop = FN / total_enfermos if total_enfermos > 0 else 0
    VN_prop = VN / total_sanos if total_sanos > 0 else 0
    FP_prop = FP / total_sanos if total_sanos > 0 else 0
    
    return VP_prop, FN_prop, VN_prop, FP_prop


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


    print(f"Precisión (Precision): {precision:.2f}")
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

def main():
    # Cargar datos de prueba
    carpeta_prueba = "/Users/nicolasfranke/Downloads/chest_xray/test/all"

    lista_jasons = [('4_0.0001_top1.json', 4),('8_0.0001_top1.json',8),
                    ('16_0.0001_top1.json',16),('32_0.0001_top1.json',32),
                    ('64_0.0001_top1.json',64),('128_0.0001_top1.json',128),
                    ('256_0.0001_top1.json',256),('512_0.0001_top1.json',512)]

    for jason, escala in lista_jasons:
        images_test, d_test = cargar_datos(carpeta_prueba, escala)
        # Cargar pesos óptimos desde el archivo JSON
        with open(f'mejorEntrenamiento_escala_alpha/{jason}', 'r') as archivo_json:
            valores_dict = json.load(archivo_json)
            w_estrella = np.array(valores_dict["w_estrella"])
            b_estrella = np.array(valores_dict["b_estrella"])

        # Realizar predicciones
        predicciones = predecir(images_test, w_estrella, b_estrella)

        # Generar matriz de confusión
        VP, FN, VN, FP = generar_matriz_confusion(d_test, predicciones)

        # Analizar efectividad y visualizar resultados
        analizar_efectividad(VP, FN, VN, FP)

if __name__ == "__main__":
    main()
