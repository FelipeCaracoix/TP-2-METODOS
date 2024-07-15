#Caracoix, Marsili, Wolodarsky
import json
import numpy as np
from functions import cargar_datos, balancear_datos, gradiente_descendente, error_cuadratico_medio, plot_error_curve

######################### PONER DIRECTORIO ####################################

def main():
    # 4x4s177, 8x8s79, 16x16s171, 32x32s150, 64x64s123, 128x128s105, 256x256s42
    array = [(4,177),(8,79),(16,171),(32,150),(64,123),(128,105),(256,42)]
    seeds = range(1,10)
    for seed in seeds:
        alpha_values = [0.0000001,0.000001,0.00001,0.0001, 0.001, 0.01]
        for alpha in alpha_values:
            for Esc in array:
                escala = Esc[0]
                images, d = cargar_datos("/Users/felip/OneDrive/Escritorio/chest_xray/train/ALL", escala)#Poner path a una carpeta con todas las immagenes de train
                images_test, d_test =  cargar_datos("/Users/felip/OneDrive/Escritorio/chest_xray/test/ALL", escala)
                np.random.seed(seed)
                b = np.random.randn(1)
                w = np.random.randn(images[0].shape[0])
                print("imagenes cargadas")
                images_balanceadas, d_balanceado = balancear_datos(images, d)
                w_estrella, b_estrella, errores_test, errores_train = gradiente_descendente(w, b, images_balanceadas, d_balanceado, alpha,images_test, d_test)
                """
                w_estrella = w_estrella.tolist()
                b_estrella = b_estrella.tolist()
                valores_dict = {
                    "w_estrella": w_estrella,
                    "b_estrella": b_estrella
                }
                """
                plot_error_curve(errores_train, alpha, "train", escala, seed)
                plot_error_curve(errores_test, alpha, "test", escala, seed)
                """
                nombre = "entrenamiento" + str(alpha_values[0]) + "_1.json"
                with open(nombre, "w") as archivo_json:
                    json.dump(valores_dict, archivo_json)
                """
                
                print(f"Valor inicial de b: {b}")
                print(f"Valor inicial de w: {w}")

if __name__ == "__main__":
    main()