import json
import numpy as np
from functions import cargar_datos, balancear_datos, gradiente_descendente, error_cuadratico_medio, plot_error_curve

#wolo
#images, d = cargar_datos("/Users/nicolasfranke/Desktop/DITELLA/Métodos Computacionales/TPs/chest_xray/test/ALL", escala=128)
#felo
#images, d = cargar_datos("/Users/felip/OneDrive/Escritorio/chest_xray/train/ALL", escala=128)
#luli-capa:
images, d = cargar_datos("/Users/victoriamarsili/Downloads/chest_xray/test/ALL", escala=128)

def main():
    #np.random.seed(42)
    b = np.random.randn(1)
    w = np.random.randn(images[0].shape[0])
    alpha_values = [0.0001, 0.01, 0.05, 0.1, 0.5]

    images_balanceadas, d_balanceado = balancear_datos(images, d)
    w_estrella, b_estrella = gradiente_descendente(w, b, images_balanceadas, d_balanceado, alpha_values[0])

    w_estrella = w_estrella.tolist()
    b_estrella = b_estrella.tolist()
    valores_dict = {
        "w_estrella": w_estrella,
        "b_estrella": b_estrella
    }

    nombre = "BW" + str(alpha_values[0]) + "___1.json"
    with open(nombre, "w") as archivo_json:
        json.dump(valores_dict, archivo_json)
    images, d = cargar_datos("/Users/victoriamarsili/Downloads/chest_xray/test/ALL", escala=128)
    errors = error_cuadratico_medio(images, w_estrella, b_estrella, d)
    print(f"Valor inicial de b: {b}")
    print(f"Valor inicial de w: {w}")

    plot_error_curve(errors, alpha_values[0])
    

if __name__ == "__main__":
    main()