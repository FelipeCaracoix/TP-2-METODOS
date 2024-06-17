import numpy as np
from derivada_b import derivadaB 
from derivada_w import derivadaW 

def f_w_b(i, w, b):
    # Calculamos el producto punto de w e i, sumamos el escalar b,
    # aplicamos la función hiperbólica tangente, sumamos 1 y dividimos por 2
    return (np.tanh(np.dot(w, i) + b) + 1) / 2

def suma_f_w_b(i_matriz, w, b, d_array):
    # Inicializamos la suma total
    suma_totalW = np.zeros(3)
    suma_totalB = 0
    # Iteramos sobre las columnas de la matriz i
    for j in range(i_matriz.shape[1]):
        # Calculamos f_w_b para la columna actual y restamos d_array[j]
        suma_totalW += derivadaW(b,d_array,i_matriz[:, j], w)
        suma_totalB += derivadaB(b,d_array,i_matriz[:, j], w)
    
    return np.append(suma_totalW, suma_totalB)