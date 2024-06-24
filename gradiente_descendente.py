import numpy as np
from formula import suma_derivada, suma_f

MAX_ITER = 1000
TOLERANCIA = 0.0001

def gradiente_descendente(w_inicial, b_inicial, i, d, alpha):
    w = w_inicial
    b = b_inicial

    iter = 0

    while iter <= MAX_ITER:
        _, (grad_w, grad_b) = suma_derivada(i, w, b, d)
        # Regla de actualización
        w_siguiente = w - alpha * grad_w
        b_siguiente = b - alpha * grad_b

        if np.linalg.norm(w_siguiente - w) < TOLERANCIA and abs(b_siguiente - b) < TOLERANCIA:
            break

        w = w_siguiente
        b = b_siguiente
        iter += 1

    return w, b
