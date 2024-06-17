import matplotlib.pyplot as plt
import numpy as np
from formula import suma_derivada, suma_f

# Máxima cantidad de iteraciones (previene loops infinitos)
MAX_ITER = 1000

# Criterio de convergencia (identifica un "plateau")
TOLERANCIA = 0.0001
def gradiente_descendente(w_inicial, b_inicial, i_matriz, d_array):
    # Valores iniciales de w y b
    w = w_inicial
    b = b_inicial

    # Factor de escala de la derivada de f (hiperparámetro)
    alpha = 0.1

    # Mientras no hayamos llegado al maximo de iteraciones
    iter = 0

    while iter <= MAX_ITER:
        # Computamos siguiente w y b a partir de la derivada de la función
        _, (grad_w, grad_b) = suma_derivada(i_matriz, w, b, d_array)
        
        w_siguiente = w - alpha * grad_w
        b_siguiente = np.array([b - alpha * grad_b])

        # Chequeamos si ya alcanzamos la convergencia
        if np.linalg.norm(w_siguiente - w) < TOLERANCIA and abs(b_siguiente - b) < TOLERANCIA:
            break

        # Preparamos la siguiente iteración
        w = w_siguiente
        b = b_siguiente
        iter += 1

    return w, b

# Función a optimizar f(w, b) 
def f(i_matriz, w, b, d_array):
    
    suma_w, suma_b = suma_derivada(i_matriz, w, b, d_array)[0]
    return suma_w.sum() + suma_b  # Sumamos los valores de f(w) y f(b)

# Función principal
def main():
    w_inicial = np.random.randn(3)
    b_inicial = np.random.randn(1)
    i_matriz = np.random.randn(3, 5)  # Ejemplo con 5 muestras
    d_array = np.random.randn(5)

    w_optimo, b_optimo = gradiente_descendente(w_inicial, b_inicial, i_matriz, d_array)
    print('w óptimo:', w_optimo)
    print('b óptimo:', b_optimo)

if __name__ == '__main__':
    main()
