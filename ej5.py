#Caracoix, Marsili, Wolodarsky
from functions import *
import time

'''
Ejercicio 5. ¿Como impacta el tamano del escalado de las im ́agenes en
la efectividad del metodo? ¿Y en el tiempo de computo?.
Realizar los experimentos y graficos acordes para estudiar estas limitaciones.
'''

valores_escala = [4,8,16,32,64,128,256,512]

for escala in valores_escala:
    print(f"Procesando escala: {escala}")

    print('Comenzando Precarga.')
    # Medir el tiempo de carga y escalado de las imágenes
    start_carga = time.perf_counter()
    images_test, d_test = cargar_datos("/Users/felip/OneDrive/Escritorio/chest_xray/test/ALL", escala)#Poner path a una carpeta con todas las immagenes de test
    images, d = cargar_datos("/Users/felip/OneDrive/Escritorio/chest_xray/train/ALL", escala)#Poner path a una carpeta con todas las immagenes de train
    end_carga = time.perf_counter()
    tiempo_carga = end_carga - start_carga
    print(f"Tiempo de carga y escalado: {tiempo_carga:.3f} segundos")

    mejor_error = float('inf')
    mejor_errores = []
    mejor_tiempo_convergencia = 0
    mejor_cant_iter = 0
    mejor_seed = 0
    alpha = 0.00001

    for seed in range(0,20):
        print(seed)
        np.random.seed(seed)
        b_ = np.random.randn(1)
        w_ = np.random.randn(images[0].shape[0])

        images_balanceadas, d_balanceado = balancear_datos(images, d)
        images_test, d_test = balancear_datos(images_test, d_test)

        start = time.perf_counter()
        w_estrella, b_estrella, cant_iter = gradiente_descendente(w_, b_, images_balanceadas, d_balanceado, alpha)
        end = time.perf_counter()
        tiempo_convergencia = end - start
        errores = error_cuadratico_medio(images_test, w_estrella, b_estrella, d_test)

        if errores[-1] < 0.239:
            mejor_error = errores[-1]
            mejor_errores = errores
            mejor_tiempo_convergencia = tiempo_convergencia
            mejor_cant_iter = cant_iter
            mejor_seed = seed

            plt.figure(figsize=(16, 10))
            plt.plot(mejor_errores, label='Error', color='b', linestyle='-', marker='o', markersize=4)
            plt.xlabel('Número de Iteraciones', fontsize=14)
            plt.ylabel('Error Cuadrático Medio', fontsize=14)
            plt.suptitle(f'Escala de las Imágenes: {escala}x{escala}', fontsize=18)
            plt.title(f'Tiempo de carga: {round(tiempo_carga, 3)}s, Tiempo de convergencia: {round(mejor_tiempo_convergencia, 3)}s, Iteraciones: {mejor_cant_iter}, Mejor error: {mejor_error}', fontsize=12)
            plt.legend(loc='upper right', fontsize=12)
            plt.grid(True)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.tight_layout()

            # Guarda el gráfico sin mostrarlo en una ventana emergente
            nombre = f"e_{escala}_a_{alpha}s_{mejor_seed}.png"
            plt.savefig(os.path.join('pruebas_con_escalas', nombre))

            # Cierra la figura actual
            plt.close()

