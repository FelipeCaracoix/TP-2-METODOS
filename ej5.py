from functions import *
import time

'''
Ejercicio 5. ¿Como impacta el tamano del escalado de las im ́agenes en
la efectividad del metodo? ¿Y en el tiempo de computo?.
Realizar los experimentos y graficos acordes para estudiar estas limitaciones.
'''
#4,8,16,32,64,128,256,512,
valores_escala = [1024]

for escala in valores_escala:
    print(f"Procesando escala: {escala}")

    print('Comenzando Precarga.')
    # Medir el tiempo de carga y escalado de las imágenes
    start_carga = time.perf_counter()
    images_test, d_test = cargar_datos("/Users/nicolasfranke/Downloads/chest_xray/test/all", escala)
    images, d =           cargar_datos("/Users/nicolasfranke/Downloads/chest_xray/train/all", escala)
    end_carga = time.perf_counter()
    tiempo_carga = end_carga - start_carga
    print(f"Tiempo de carga y escalado: {tiempo_carga:.3f} segundos")

    np.random.seed(42)
    b_ = np.random.randn(1)
    w_ = np.random.randn(images[0].shape[0])

    images_balanceadas, d_balanceado = balancear_datos(images, d)
    images_test, d_test = balancear_datos(images_test, d_test)

    start = time.perf_counter()
    w_estrella, b_estrella, cant_iter = gradiente_descendente(w_, b_, images_balanceadas, d_balanceado, 0.0001)
    end = time.perf_counter()
    tiempo_convergencia = end - start
    errores = error_cuadratico_medio(images_test, w_estrella, b_estrella, d_test)

    plt.figure(figsize=(16, 10))
    plt.plot(errores, label='Error', color='b', linestyle='-', marker='o', markersize=4)
    plt.xlabel('Número de Iteraciones', fontsize=14)
    plt.ylabel('Error Cuadratico Medio', fontsize=14)
    plt.suptitle(f'Escala de las Imagenes: {escala}x{escala}', fontsize=18)
    plt.title(f'Tiempo de carga: {round(tiempo_carga, 3)}s, Tiempo de convergencia: {round(tiempo_convergencia, 3)}s, Iteraciones: {cant_iter}, Mejor error:{errores[-1]}',fontsize=12)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    # Guarda el gráfico sin mostrarlo en una ventana emergente
    nombre = f"escala_{escala}.png"
    plt.savefig("pruebas_con_escalas/" + nombre)

    # Cierra la figura actual
    plt.close()