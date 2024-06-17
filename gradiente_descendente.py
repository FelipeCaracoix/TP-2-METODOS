import matplotlib.pyplot as plt
import numpy as np

# Máxima cantidad de iteraciones (previene loops infinitos)
MAX_ITER = 1000

# Criterio de convergencia (identifica un "plateau")
TOLERANCIA = 0.0001

# Derivada de f(x)
def df(x):
    return 2*x

# Función de búsqueda del mínimo de f(x)
def gradiente_descendente(x_0):
    # Valor inicial de x
    x_t = x_0

    # Factor de escala de la derivada de f (hiperparámetro)
    alpha = 0.1

    # Mientras no hayamos llegado al maximo de iteraciones
    iter = 0

    while iter <= MAX_ITER:
        print("Iteración: ", iter, "- Mínimo alcanzado hasta el momento: ", f(x_t))
        # Computamos siguiente x a partir de la derivada de la función
        x_tsig = x_t - alpha * df(x_t)

        # Chequeamos si ya alcanzamos la convergencia
        if abs(f(x_tsig) - f(x_t)) < TOLERANCIA:
            break

        # Preparamos la siguiente iteración
        x_t = x_tsig
        iter = iter + 1

    return x_tsig

# Función a optimizar f(x) = x**2
def f(x):
    return x**2