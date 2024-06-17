"""
Sample code automatically generated on 2024-06-16 22:15:22

by www.matrixcalculus.org

from input

Fwb = 0.5 * (np.tanh((i * w) + b) + 1)

d Fwb/d w = 0.5* ( 1 - np.tanh^2((w).dot(i) + b)).dot(i))
d Fwb/d b = 0.5* ( 1 - np.tanh^2((w).dot(i) + b))

sum = sum( (0.5 * np.tanh((i * w) + b) + 1) - di)^2

d sum/d w = 2 * sum(0.5 * (np.tanh((i * w) + b) + 1) - di) * 0.5* ( 1 - np.tanh^2 ((w).dot(i) + b) ).dot(i))
d sum/d b = 2 * sum(0.5 * (np.tanh((i * w) + b) + 1) - di) * 0.5* ( 1 - np.tanh^2 ((w).dot(i) + b) )

d/dw sum( ((0.5 * tanh((i * w) + vector(b)) + vector(1)) - vector(d)).^2) 
        = i'*((vector(1)+0.5*tanh(i*w+b*vector(1))-d*vector(1)).*(vector(1)-tanh(i*w+b*vector(1)).^2))

where

b is a scalar
d is a scalar
i is a matrix
w is a vector

The generated code is provided "as is" without warranty of any kind.
"""

from __future__ import division, print_function, absolute_import

import numpy as np

def fAndG(b, d, i, w):
    if isinstance(b, np.ndarray):
        dim = b.shape
        assert dim == (1, )
    if isinstance(d, np.ndarray):
        dim = d.shape
        assert dim == (1, )
    assert isinstance(i, np.ndarray)
    dim = i.shape
    assert len(dim) == 2
    i_rows = dim[0]
    i_cols = dim[1]
    assert isinstance(w, np.ndarray)
    dim = w.shape
    assert len(dim) == 1
    w_rows = dim[0]
    assert w_rows == i_cols

    t_0 = np.tanh(((i).dot(w) + (b * np.ones(i_rows))))
    t_1 = ((np.ones(i_rows) + (0.5 * t_0)) - (d * np.ones(i_rows)))
    functionValue = np.sum((t_1 ** 2))
    gradient = (i.T).dot((t_1 * (np.ones(i_rows) - (t_0 ** 2))))

    return functionValue, gradient

def checkGradient(b, d, i, w):
    # numerical gradient checking
    # f(x + t * delta) - f(x - t * delta) / (2t)
    # should be roughly equal to inner product <g, delta>
    t = 1E-6
    delta = np.random.randn(3)
    f1, _ = fAndG(b, d, i, w + t * delta)
    f2, _ = fAndG(b, d, i, w - t * delta)
    f, g = fAndG(b, d, i, w)
    print('approximation error',
          np.linalg.norm((f1 - f2) / (2*t) - np.tensordot(g, delta, axes=1)))

"""def generateRandomData():
    b = np.random.randn(1)
    d = np.random.randn(1)
    i = np.random.randn(3, 3)
    w = np.random.randn(3)

    return b, d, i, w

if __name__ == '__main__':
    b, d, i, w = generateRandomData()
    functionValue, gradient = fAndG(b, d, i, w)
    print('functionValue = ', functionValue)
    print('gradient = ', gradient)

    print('numerical gradient checking ...')
    checkGradient(b, d, i, w)"""


############################################################
"""
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
"""