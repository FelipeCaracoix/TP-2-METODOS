from functions import *
import time
alpha_values = [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 3]

images, d = cargar_datos("/Users/felip/OneDrive/Escritorio/chest_xray/train/ALL", escala=128)
images_test, d_test = cargar_datos("/Users/felip/OneDrive/Escritorio/chest_xray/test/ALL", escala=128)

b = np.random.randn(1)
w = np.random.randn(images[0].shape[0])

images_balanceadas, d_balanceado = balancear_datos(images, d)
images_test, d_test = balancear_datos(images_test, d_test)

for alpha in alpha_values:

    start = time.perf_counter()
    w_estrella, b_estrella, cant_iter = gradiente_descendente(w, b, images_balanceadas, d_balanceado, alpha_values[0])
    end = time.perf_counter()
    tiempo = end - start
    errores = error_cuadratico_medio(images_test, w_estrella, b_estrella, d_test)
    plot_error_curve(errores, alpha)
    plt.title("Tiempo en converger: " + str(round(tiempo, 3)) + "s.  Iteraciones: "+ str(cant_iter), fontsize=12)
    # Guarda el gráfico sin mostrarlo en una ventana emergente
    nombre = f"1plot_test_nuevo1_{alpha}.png"
    plt.savefig("pruebas_con_alphas/" + nombre)

    # Cierra la figura actual
    plt.close()