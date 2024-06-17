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

i = np.array(abrirImagenesEscaladas("/Users/victoriamarsili/Downloads/chest_xray/test/PNEUMONIA", escala=32))
print(i.shape)
d = np.random.randn(1024)
def balancear_datos(imagenes_entrenamiento):
    b = np.random.randn(1)
    w = np.random.randn(390)
    imagenes_entrenamiento_balanceadas = gradiente_descendente(w,b,imagenes_entrenamiento,d)
    return imagenes_entrenamiento_balanceadas

w_estrella, b_estrella = balancear_datos(i)
print(funcion_f(i,w_estrella, b_estrella, d))