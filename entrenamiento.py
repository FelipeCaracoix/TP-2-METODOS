#Caracoix, Marsili, Wolodarsky
import json
import numpy as np
from functions import cargar_datos, balancear_datos, gradiente_descendente, error_cuadratico_medio, plot_error_curve
from ej6 import matriz_confusion
######################### PONER DIRECTORIO ####################################

def main():
    # 4x4s177, 8x8s79, 16x16s171, 32x32s150, 64x64s123, 128x128s105, 256x256s42
    #array = [(4,177),(8,79),(16,171),(32,150),(64,123),(128,105),(256,42)]
    array = [(4,177),(8,79),(16,171),(32,150),(64,123),(128,105),(256,42)]
    seeds = range(1,50)
    for seed in seeds:
    #[0.0000001,0.000001,0.00001,0.0001, 0.001, 0.01]
        alpha_values = [0.0001]
        
        for alpha in alpha_values:
            for Esc in array:
                escala = Esc[0]
                seed = Esc[1]
                images, d = cargar_datos("/Users/nicolasfranke/Downloads/chest_xray/train/all", escala)#Poner path a una carpeta con todas las immagenes de train
                images_test, d_test =  cargar_datos("/Users/nicolasfranke/Downloads/chest_xray/test/all", escala)
                np.random.seed(seed)
                b = np.random.randn(1)
                w = np.random.randn(images[0].shape[0])
                print("imagenes cargadas")
                images_balanceadas, d_balanceado = balancear_datos(images, d)
                w_estrella, b_estrella, errores_test, errores_train = gradiente_descendente(w, b, images_balanceadas, d_balanceado, alpha,images_test, d_test)

                plot_error_curve(errores_train, alpha, "train", escala, seed)
                plot_error_curve(errores_test, alpha, "test", escala, seed)
                matriz_confusion(images_test,d_test, b_estrella,w_estrella,escala,alpha,seed)

                print(f"Valor inicial de b: {b}")
                print(f"Valor inicial de w: {w}")

if __name__ == "__main__":
    main()