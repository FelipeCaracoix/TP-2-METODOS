#Caracoix, Marsili, Wolodarsky
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

from derivadas import derivadaB, derivadaW

def f_w_b(b, d, i, w):
    # Calculamos el producto punto de w e i, sumamos el escalar b,
    # aplicamos la función hiperbólica tangente, sumamos 1 y dividimos por 2
    return ((np.tanh((np.dot(w, i) + b)) + 1) / 2 - d )**2

def suma_derivada(i, w, b, d_array):
    """
    Calcula la suma de las derivadas parciales con respecto a w y b.

    Args:
        i (ndarray): Matriz de imágenes vectorizadas.
        w (ndarray): Vector de pesos.
        b (float): Sesgo.
        d_array (ndarray): Vector de etiquetas (0 o 1).

    Returns:
        tuple: Suma de las derivadas parciales con respecto a w y b.
    """
    Dsuma_totalW = np.zeros(i[0].shape[0])
    Dsuma_totalB = 0
    suma_totalW = np.zeros(i[0].shape[0])
    suma_totalB = 0
    for j in range(i.shape[0]):
        derW = derivadaW(b,d_array[j],i[j], w)
        derB = derivadaB(b,d_array[j],i[j], w)
        suma_totalW += derW[0]
        suma_totalB += derB[0]
        Dsuma_totalW += derW[1]

        Dsuma_totalB += derB[1]

    return ((suma_totalW, suma_totalB),(Dsuma_totalW, Dsuma_totalB))
            # valor f(W), valor f(B),    valor df(W), valor df(B)

MAX_ITER = 1000
TOLERANCIA = 0.0001

def gradiente_descendente(w_inicial, b_inicial, i, d, alpha,i_test,d_test):
    """
    Optimiza los pesos y sesgo utilizando el método de descenso por gradiente.

    Args:
        w_inicial (ndarray): Vector inicial de pesos.
        b_inicial (float): Sesgo inicial.
        i (ndarray): Matriz de imágenes vectorizadas.
        d (ndarray): Vector de etiquetas (0 o 1).
        alpha (float): Tasa de aprendizaje.

    Returns:
        tuple: Vector de pesos y sesgo optimizados.
    """
    w = w_inicial
    b = b_inicial
    errores_test = []
    errores_train = []
    iter = 0

    while iter <= MAX_ITER:
        _, (grad_w, grad_b) = suma_derivada(i, w, b, d)
        # Regla de actualización
        errores_test.append(error_cuadratico_medio(i_test,w,b,d_test))
        errores_train.append(error_cuadratico_medio(i,w,b,d))
        w_siguiente = w - alpha * grad_w
        b_siguiente = b - alpha * grad_b
        print(iter)

        if np.linalg.norm(w_siguiente - w) < TOLERANCIA and abs(b_siguiente - b) < TOLERANCIA:
            break

        w = w_siguiente
        b = b_siguiente
        iter += 1

    return w, b, errores_test, errores_train

def abrirImagenesEscaladas(carpeta, escala):
    """
    Carga y escala imágenes desde una carpeta específica.

    Args:
        carpeta (str): Ruta de la carpeta que contiene las imágenes.
        escala (int): Tamaño al que se escalarán las imágenes.

    Returns:
        tuple: Matriz de imágenes vectorizadas y vector de etiquetas.
    """
    imagenes = []
    etiquetas = []
    extensiones_validas = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')

    for dirpath, dirnames, filenames in os.walk(carpeta):
        for file in filenames:
            if file.lower().endswith(extensiones_validas):
                img = Image.open(os.path.join(carpeta, file))
                img = img.resize((escala, escala))
                img = img.convert('L')  # Convertir a escala de grises
                img = np.asarray(img).reshape((escala**2)) / 255.0
                imagenes.append(img)
                if "virus" in file or "bacteria" in file:
                    etiqueta = 1
                else:
                    etiqueta = 0
                etiquetas.append(etiqueta)
    return np.array(imagenes), np.array(etiquetas)

def cargar_datos(carpeta_base, escala):
    """
    Carga las imágenes y etiquetas desde la carpeta base.

    Args:
        carpeta_base (str): Ruta de la carpeta que contiene las imágenes.
        escala (int): Tamaño al que se escalarán las imágenes.

    Returns:
        tuple: Matriz de imágenes vectorizadas y vector de etiquetas.
    """
    images, labels = abrirImagenesEscaladas(carpeta_base, escala)
    return images, labels

def balancear_datos(imagenes, etiquetas):
    """
    Balancea el conjunto de datos para tener una cantidad equitativa de imágenes con y sin neumonía.

    Args:
        imagenes (ndarray): Matriz de imágenes vectorizadas.
        etiquetas (ndarray): Vector de etiquetas (0 o 1).

    Returns:
        tuple: Matriz de imágenes balanceadas y vector de etiquetas balanceadas.
    """
    # Separar las imágenes en dos clases
    class_0 = [img for img, label in zip(imagenes, etiquetas) if label == 0]
    class_1 = [img for img, label in zip(imagenes, etiquetas) if label == 1]

    # Determinar la cantidad mínima de muestras entre las clases
    n_samples = min(len(class_0), len(class_1))

    # Seleccionar n_samples de cada clase
    balanced_class_0 = class_0[:n_samples]
    balanced_class_1 = class_1[:n_samples]

    # Combinar y mezclar las clases balanceadas
    imagenes_balanceadas = balanced_class_0 + balanced_class_1
    etiquetas_balanceadas = [0] * n_samples + [1] * n_samples
  
    return np.array(imagenes_balanceadas), np.array(etiquetas_balanceadas)

def error_cuadratico_medio(i, w, b, d_array):
    """
    Calcula el error cuadrático medio para el conjunto de datos dado.

    Args:
        i (ndarray): Matriz de imágenes vectorizadas.
        w (ndarray): Vector de pesos.
        b (float): Sesgo.
        d_array (ndarray): Vector de etiquetas (0 o 1).

    Returns:
        list: Lista de errores cuadráticos medios.
    """
    suma_total = 0.0
 
    for j in range(i.shape[0]):
        z = np.dot(w, i[j]) + b
        prediccion = (np.tanh(z) + 1) / 2
        suma_total += (prediccion - d_array[j])**2

    return suma_total / i.shape[0]
def plot_error_curve(errors, alpha, type, escala, seed):
    plt.figure(figsize=(10, 6))
    plt.plot(errors, label='Error', color='b', linestyle='-')
    plt.xlabel('Número de Iteraciones', fontsize=14)
    plt.ylabel('Error Cuadratico Medio', fontsize=14)
    plt.suptitle(f'Error Reduction Over Iterations', fontsize=16)
    plt.title(f"Alpha = {alpha}, Seed = {seed}, Min_Error = {errors[-1]}, {type}", fontsize=12)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    carpeta = f's{seed}'
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)
     # Guarda el gráfico sin mostrarlo en una ventana emergente
    nombre = f"t_{type}e_{escala}_a_{alpha}s_{seed}.png"
    plt.savefig(os.path.join(f's{seed}', nombre))

    # Cierra la figura actual
    plt.close()

def predecir(i, w, b, umbral=0.5):
    """
    Realiza predicciones utilizando el modelo entrenado.

    Args:
        i (ndarray): Matriz de imágenes vectorizadas.
        w (ndarray): Vector de pesos.
        b (float): Sesgo.
        umbral (float): Umbral de decisión para convertir probabilidades en predicciones binarias.

    Returns:
        ndarray: Vector de predicciones binarias (0 o 1).
    """
    predicciones = []
    for imagen in i:
        z = np.dot(w, imagen) + b
        prediccion = 1 if (np.tanh(z) + 1) / 2 >= umbral else 0
        predicciones.append(prediccion)

    return np.array(predicciones)

