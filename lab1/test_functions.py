import numpy as np

class Func:
    def __init__(self, f, df):
        self.f = f
        self.df = df
    
    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def d(self, *args, **kwargs):
        return self.df(*args, **kwargs)

def _f1(X):
    x, y = X[0], X[1]
    return x**2 + 3*y**2 - x*y - 3*x - y + 5

def _f1x(X):
    x, y = X[0], X[1]
    return 2*x - y - 3

def _f1y(X):
    x, y = X[0], X[1]
    return 6*y - x - 1

def _df1(X): 
    return np.array([_f1x(X), _f1y(X)])

# ------------- EXPORTS ---------------

f1 = Func(_f1, _df1)