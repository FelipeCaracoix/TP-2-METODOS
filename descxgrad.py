import numpy as np
import os
from PIL import Image
from gradiente_descendente import gradiente_descendente

def abrirImagenesEscaladas(carpeta, escala=32):
    imagenes = []
    etiquetas = []

    for dirpath, dirnames, filenames in os.walk(carpeta):
        for file in filenames:
            img = Image.open(os.path.join(carpeta, file))
            img = img.resize((escala, escala))
            img = img.convert('L')  # Convertir a escala de grises
            img = np.asarray(img).reshape((escala ** 2)) / 255

            # Asignar etiquetas basadas en el nombre del archivo
            if "virus" in file or "bacteria" in file:
                etiqueta = 1
            else:
                etiqueta = 0

            imagenes.append(img)
            etiquetas.append(etiqueta)

    return imagenes, etiquetas

def cargar_datos(carpeta_base, escala=32):
    pneumonia_path = os.path.join(carpeta_base, "PNEUMONIA")
    normal_path = os.path.join(carpeta_base, "NORMAL")

    pneumonia_images, pneumonia_labels = abrirImagenesEscaladas(pneumonia_path, escala)
    normal_images, normal_labels = abrirImagenesEscaladas(normal_path, escala)

    # Unir las imágenes y etiquetas
    X = pneumonia_images + normal_images
    y = pneumonia_labels + normal_labels

    return np.array(X), np.array(y)

def balancear_datos(imagenes_entrenamiento, etiquetas_entrenamiento):
    b = np.random.randn(1)
    w = np.random.randn(390)
    imagenes_entrenamiento_balanceadas, errores = gradiente_descendente(w, b, imagenes_entrenamiento, etiquetas_entrenamiento)
    return imagenes_entrenamiento_balanceadas, errores

# Cargar imágenes y datos
train_images, d_entrenamiento = cargar_datos("/Users/victoriamarsili/Downloads/chest_xray/train", escala=32)
test_images, d_test = cargar_datos("/Users/victoriamarsili/Downloads/chest_xray/test", escala=32)

# Inicializar d a partir de los nombres de los archivos
d_entrenamiento = np.array([1 if "virus" in fname or "bacteria" in fname else 0 for fname in d_entrenamiento])
d_test = np.array([1 if "virus" in fname or "bacteria" in fname else 0 for fname in d_test])

# Ejecutar el balanceo de datos y obtener errores
w_estrella, b_estrella, train_errors, test_errors = gradiente_descendente(
    np.random.randn(train_images.shape[1]), np.random.randn(1), train_images, d_entrenamiento, test_images, d_test)

# Graficar errores
import matplotlib.pyplot as plt

plt.plot(train_errors, label='Error de Entrenamiento')
plt.plot(test_errors, label='Error de Prueba')
plt.xlabel('Iteraciones')
plt.ylabel('Error Cuadrático Medio')
plt.title('Evolución del Error durante el Entrenamiento')
plt.legend()
plt.show()
