import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from gradiente_descendente import gradiente_descendente

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
    print(i)
    suma_total = 0.0
    errores = []
    for j in range(i.shape[0]):
        prediccion = (np.tanh(w.dot(i[j]) + b) + 1) / 2
        suma_total += (prediccion - d_array[j])**2
        if suma_total/(j+1) == float("inf"):
            errores.append(1)
        errores.append(suma_total/(j+1))

    return errores

# Cargar imágenes y datos
images, d = cargar_datos("/Users/nicolasfranke/Desktop/DITELLA/Métodos Computacionales/TPs/chest_xray/test/ALL", escala=32)

b = np.random.randn(1)
w = np.random.randn(images[0].shape[0])
alpha_values = [0.001, 0.01, 0.05, 0.1, 0.5]

images_balanceadas, d_balanceado = balancear_datos(images, d)
w_estrella, b_estrella = gradiente_descendente(w, b, images_balanceadas, d_balanceado, alpha_values[4])

errors = error_cuadratico_medio(images_balanceadas, w_estrella, b_estrella, d_balanceado)
print(errors)


plt.plot(errors, label='Error')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.title('Decreasing Error over Iterations')
plt.legend()
plt.show()