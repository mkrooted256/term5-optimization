import numpy as np

class Func:
    def __init__(self, f, df):
        self.f = f
        self.df = df
    
    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def d(self, *args, **kwargs):
        return self.df(*args, **kwargs)

    def auto_d(self, x, dx=1e-5):
        x = np.array(x).astype(float)
        d = np.empty_like(x)
        for i in range(x.size):
            d[i] = self.auto_di(i, x, dx)
        return d

    def auto_di(self, i, x, dx_i=1e-5):
        dx = np.zeros_like(x)
        dx[i] = dx_i 
        return (self.f(x+dx) - self.f(x-dx))/(2*dx_i)

# ------------- FUNCS ---------------

# --- f1 ---
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

# --- special ---

def _himmelblau(X):
    x, y =  X[0], X[1]
    return np.square(x*x + y - 11) + np.square(x + y*y -7)

def _ackley(X):
    x, y =  X[0], X[1]
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x*x + y*y))) - np.exp(0.5*(np.cos(2*3.1415*x)+np.cos(2*3.1415*y)))+ np.e + 20


# ------------- EXPORTS ---------------

f1 = Func(_f1, _df1)

himmelblau = Func(_himmelblau, None)
ackley = Func(_ackley, None)
