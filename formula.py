import numpy as np
from derivada_b import derivadaB
from derivada_w import derivadaW

def f_w_b(b, d, i, w):
    # Calculamos el producto punto de w e i, sumamos el escalar b,
    # aplicamos la función hiperbólica tangente, sumamos 1 y dividimos por 2
    return ((np.tanh((np.dot(w, i) + b)) + 1) / 2 - d )**2

def suma_derivada(i, w, b, d_array):
    # Inicializamos la suma total
    Dsuma_totalW = np.zeros(i[0].shape[0])
    Dsuma_totalB = 0
    suma_totalW = np.zeros(i[0].shape[0])
    suma_totalB = 0
    # Iteramos sobre las columnas de la matriz i
    for j in range(i.shape[0]):
        # Calculamos f_w_b para la columna actual y restamos d_array[j]
        derW = derivadaW(b,d_array[j],i[j], w)
        derB = derivadaB(b,d_array[j],i[j], w)
        suma_totalW += derW[0]
        suma_totalB += derB[0]
        Dsuma_totalW += derW[1]

        Dsuma_totalB += derB[1]

    return ((suma_totalW, suma_totalB),(Dsuma_totalW, Dsuma_totalB))
            # valor f(W), valor f(B),    valor df(W), valor df(B)

def suma_f(i, w, b, d_array):
    # Inicializamos la suma total
    suma_total = 0
   
    # Iteramos sobre las columnas de la matriz i
    for j in range(i.shape[0]):
        # Calculamos f_w_b para la columna actual y restamos d_array[j]
        suma_total += f_w_b(b,d_array[j],i[j], w)
    return suma_total