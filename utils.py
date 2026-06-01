import numpy as np
from collections.abc import Iterable
from numba import njit, float64

def flatten(obj):
    def flat(x):
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for item in x:
                yield from flat(item)
        else:
            yield x
    return np.fromiter(flat(obj), dtype=float)

@njit(float64(float64, float64))
def logsumexp2(a, b):
    m = a if a > b else b
    return m + np.log(np.exp(a - m) + np.exp(b - m))

@njit(float64(float64, float64, float64, float64))
def logsumexp4(a, b, c, d):
    m = a
    if b > m: m = b
    if c > m: m = c
    if d > m: m = d

    s = np.exp(a - m) + np.exp(b - m) + np.exp(c - m) + np.exp(d - m)
    return m + np.log(s)