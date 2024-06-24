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

def balancear_datos(imagenes, diagnostico):
    imagenes_balanceadas = gradiente_descendente(w, b, imagenes, diagnostico)
    return imagenes_balanceadas

def error_cuadratico_medio(i, w, b, d_array):
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
#print(balancear_datos(images,d))


errors = error_cuadratico_medio(images,w,b,d)
print(errors)


plt.plot(errors, label='Error')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.title('Decreasing Error over Iterations')
plt.legend()
plt.show()