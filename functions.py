import numpy as np

def fAndG(b, d, i, w):
    assert isinstance(b, float) or isinstance(b, int)
    assert isinstance(d, float) or isinstance(d, int)
    assert isinstance(i, np.ndarray) and i.ndim == 2
    assert isinstance(w, np.ndarray) and w.ndim == 1
    
    num_columns_i = i.shape[1]  # número de columnas en la matriz i

    Fwb = (np.tanh((i * w) + b) + 1)/2

    functionValue = 0
    gradient = np.zeros_like(w, dtype=np.float64)  # inicializa gradient con tipo float64
    
    for col in range(num_columns_i):
        i_col = i[:, col]  # selecciona la columna específica de i
        
        t_0 = np.tanh((i_col * w) + b)
        t_1 = (1 + 0.5 * t_0) - d
        
        functionValue += np.sum(t_1 ** 2)
        gradient += t_1 * (1 - t_0 ** 2) * i_col
    
    return functionValue, gradient

def generateRandomData():
    b = 2
    d = 3
    i = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    w = np.array([1, 1, 1])

    return b, d, i, w

if __name__ == '__main__':
    b, d, i, w = generateRandomData()
    print("Matriz i:")
    print(i)
    
    functionValue, gradient = fAndG(b, d, i, w)
    print('\nfunctionValue = ', functionValue)
    print('gradient = ', gradient)
