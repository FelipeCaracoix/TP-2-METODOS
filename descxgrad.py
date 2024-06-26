import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from gradiente_descendente import gradiente_descendente

def abrirImagenesEscaladas(carpeta, escala=32):
    imagenes = []
    etiquetas = []
    extensiones_validas = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')

    for dirpath, dirnames, filenames in os.walk(carpeta):
        for file in filenames:
            if file.lower().endswith(extensiones_validas):
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
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('Error Cuadratico Medio', fontsize=14)
    plt.title(f'Error Reduction Over Iterations\n(Alpha = {alpha}, Numero de Imagenes = {len(errors)})', fontsize=16)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

# Cargar imágenes y datos
#wolo
print('hola')
images, d = cargar_datos("/Users/nicolasfranke/Desktop/DITELLA/Métodos Computacionales/TPs/chest_xray/test/ALL", escala=128)
print('hola2')
#felo
# images, d = cargar_datos("/Users/felip/OneDrive/Escritorio/chest_xray/train/ALL", escala=32)
# #luli-capa:
# images, d = cargar_datos("/Users/victoriamarsili/Downloads/chest_xray/test/ALL", escala=32)
#b = np.random.uniform(0,1,1)
# w = np.random.randn(images[0].shape[0])
# w = np.array([-0.1]*images[0].shape[0])

for r in range(60,80):
    np.random.seed(r)

    b = np.random.randn(1)
    w = np.random.randn(images[0].shape[0])
    print(b,w)

    alpha_values = [0.001, 0.01, 0.05, 0.1, 0.5]

    images_balanceadas, d_balanceado = balancear_datos(images, d)
    w_estrella, b_estrella = gradiente_descendente(w, b, images_balanceadas, d_balanceado, 0.0001)

    errors = error_cuadratico_medio(images_balanceadas, w_estrella, b_estrella, d_balanceado)
    print("seed: ",r)
    plot_error_curve(errors,0.0001)
