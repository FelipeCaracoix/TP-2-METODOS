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
    # Inicializamos la suma total
    Dsuma_totalW = np.zeros(i[0].shape[0])
    Dsuma_totalB = 0
    suma_totalW = np.zeros(i[0].shape[0])
    suma_totalB = 0
    # Iteramos sobre las columnas de la matriz i
    for j in range(i.shape[0]):
        # Calculamos f_w_b para la columna actual y restamos d_array[j]
        derW = derivadaW(b,d_array[j],i[j], w)
        derB = derivadaB(b,d_array[j],i[j], w)
        suma_totalW += derW[0]
        suma_totalB += derB[0]
        Dsuma_totalW += derW[1]

        Dsuma_totalB += derB[1]

    return ((suma_totalW, suma_totalB),(Dsuma_totalW, Dsuma_totalB))
            # valor f(W), valor f(B),    valor df(W), valor df(B)

MAX_ITER = 10000
TOLERANCIA = 0.0001

def gradiente_descendente(w_inicial, b_inicial, i, d, alpha):
    w = w_inicial
    b = b_inicial

    iter = 0

    while iter <= MAX_ITER:
        print(iter)
        _, (grad_w, grad_b) = suma_derivada(i, w, b, d)
        # Regla de actualización
        w_siguiente = w - alpha * grad_w
        b_siguiente = b - alpha * grad_b

        if np.linalg.norm(w_siguiente - w) < TOLERANCIA and abs(b_siguiente - b) < TOLERANCIA:
            break

        w = w_siguiente
        b = b_siguiente
        print(b_siguiente)
        iter += 1

    return w, b



def abrirImagenesEscaladas(carpeta, escala=32):
    imagenes = []
    etiquetas = []

    for dirpath, dirnames, filenames in os.walk(carpeta):
        for file in filenames:
            img = Image.open( os.path.join(carpeta, file) )
            img = img.resize((escala, escala))
            img.convert('1')
            img = np.asarray(img)
            if len(img.shape)==3:
                img = img[:,:,0].reshape((escala**2 )) / 255
            else:
                img = img.reshape((escala**2 )) / 255

            imagenes.append( img )
            # Asignar etiquetas basadas en el nombre del archivo
            if "virus" in file or "bacteria" in file:
                etiqueta = 1

            else:
                etiqueta = 0

            etiquetas.append(etiqueta)
    return np.array(imagenes), np.array(etiquetas)

def cargar_datos(carpeta_base, escala=32):
    images, labels = abrirImagenesEscaladas(carpeta_base, escala)
    return images, labels

def balancear_datos(imagenes, etiquetas):
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

    combined = list(zip(imagenes_balanceadas, etiquetas_balanceadas))
    np.random.shuffle(combined)

    imagenes_balanceadas, etiquetas_balanceadas = zip(*combined)

    return np.array(imagenes_balanceadas), np.array(etiquetas_balanceadas)

def error_cuadratico_medio(i, w, b, d_array):
    suma_total = 0.0
    errores = []
    for j in range(i.shape[0]):
        z = np.dot(w, i[j]) + b
        print(z)
        prediccion = (np.tanh(z) + 1) / 2
        print(prediccion, d_array[j])
        #prediccion = (np.tanh(w.dot(i[j]) + b) + 1) / 2
        suma_total += (prediccion - d_array[j])**2
        if suma_total/(j+1) == float("inf"):
            errores.append(1)
        errores.append(suma_total/(j+1))

    return errores


def plot_error_curve(errors, alpha):
    plt.figure(figsize=(10, 6))
    plt.plot(errors, label='Error', color='b', linestyle='-', marker='o', markersize=4)
    plt.xlabel('Numero de Imagenes', fontsize=14)
    plt.ylabel('Error Cuadratico Medio', fontsize=14)
    plt.title(f'Error Reduction Over Iterations\n(Alpha = {alpha}, Numero de Imagenes = {len(errors)})', fontsize=16)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show() 
