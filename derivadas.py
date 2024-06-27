"""
Derivada de B
d/db ((tanh(w'*i+b)+1)/2-d)^2 = (1-tanh(b+w'*i).^2)*((1+tanh(b+w'*i))/2-d)

where

b is a scalar
d is a scalar
i is a vector
w is a vector
"""

from __future__ import division, print_function, absolute_import

import numpy as np

def derivadaB(b, d, i, w):
    if isinstance(b, np.ndarray):
        dim = b.shape
        assert dim == (1, )
    if isinstance(d, np.ndarray):
        dim = d.shape
        assert dim == (1, )
    assert isinstance(i, np.ndarray)
    dim = i.shape
    assert len(dim) == 1
    i_rows = dim[0]
    assert isinstance(w, np.ndarray)
    dim = w.shape
    assert len(dim) == 1
    w_rows = dim[0]
    assert i_rows == w_rows

    t_0 = np.tanh((b + (w).dot(i)))
    t_1 = (((1 + t_0) / 2) - d)
    functionValue = (t_1 ** 2)
    gradient = ((1 - (t_0 ** 2)) * t_1)

    return functionValue, gradient


"""
Derivada de W
d/dw ((tanh(w'*i+b)+1)/2-d)^2 = (1-tanh(b+w'*i).^2)*((1+tanh(b+w'*i))/2-d)*i

where

b is a scalar
d is a scalar
i is a vector
w is a v