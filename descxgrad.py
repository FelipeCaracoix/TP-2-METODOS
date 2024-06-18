import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from gradiente_descendente import gradiente_descendente, funcion_f

def abrirImagenesEscaladas( carpeta, escala=32 ):
    # abre todas las imagenes de la carpeta, y las escala de tal forma que midan (escala x escala)px
    # devuelve las imagenes aplanadas -> vectores de tamano escala^2 con valores entre 0 y 1
    imagenes = []

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

    return imagenes

# Función para cargar imágenes y etiquetas desde una carpeta dada
def cargar_datos(carpeta_base, escala=32):
    pneumonia_path = os.path.join(carpeta_base, "PNEUMONIA")
    normal_path = os.path.join(carpeta_base, "NORMAL")

    # Cargar imágenes y etiquetas
    pneumonia_images = abrirImagenesEscaladas(pneumonia_path, escala)
    normal_images = abrirImagenesEscaladas(normal_path, escala)

    X = np.hstack((pneumonia_images, normal_images))
    y = np.array([1] * pneumonia_images.shape[1] + [0] * normal_images.shape[1])

    return X, y

d = np.random.randn(1024)
def balancear_datos(imagenes_entrenamiento):
    b = np.random.randn(1)
    w = np.random.randn(390)
    imagenes_entrenamiento_balanceadas = gradiente_descendente(w,b,imagenes_entrenamiento,d)
    return imagenes_entrenamiento_balanceadas

#w_estrella, b_estrella = balancear_datos(i)
#print(funcion_f(i,w_estrella, b_estrella, d))

# Cargar imágenes y datos
train_images, d_entrenamiento = cargar_datos("/Users/victoriamarsili/Downloads/chest_xray", escala=32)
test_images, d_test = cargar_datos("/path/to/dataset/test", escala=32)

# Ejecutar el balanceo de datos y obtener errores
w_estrella, b_estrella, train_errors, test_errors = gradiente_descendente(
    np.random.randn(train_images.shape[0]), np.random.randn(1), train_images, d_entrenamiento, test_images, d_test)

# Graficar errores
plt.plot(train_errors, label='Error de Entrenamiento')
plt.plot(test_errors, label='Error de Prueba')
plt.xlabel('Iteraciones')
plt.ylabel('Error Cuadrático Medio')
plt.title('Evolución del Error durante el Entrenamiento')
plt.legend()
plt.show()