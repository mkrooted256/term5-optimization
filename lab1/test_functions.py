import numpy as np

class Func:
    def __init__(self, f, df=None, ddf=None):
        self.f = f
        self.df = df
        self.ddf = ddf
    
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

def _ddf1(X):
    x, y = X[0], X[1]
    return np.array([
        [2, -1],
        [-1, 6]
    ])


# --- special ---

def _himmelblau(X):
    x, y =  X[0], X[1]
    return np.square(x*x + y - 11) + np.square(x + y*y -7)

def _d_himmelblau(X):
    x, y =  X[0], X[1]
    return np.array([
        x * (4 * x**2 - 42) + y * (4 *x + 2* y) - 14,
        x * (2*x + 4*y) + y*(4*y**2 - 26) - 22
    ])

def _dd_himmelblau(X):
    x, y =  X[0], X[1]
    return np.array([
        [-42 + 12*x**2 + 4*y, 4*(x+y)],
        [4*(x+y), -26 + 4*x + 12*y**2]
    ])

# ---

def _ackley(X):
    x, y =  X[0], X[1]
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x*x + y*y))) - np.exp(0.5*(np.cos(2*3.1415*x)+np.cos(2*3.1415*y)))+ np.e + 20

def _d_ackley(X):
    x, y =  X[0], X[1]
    return np.array([
        (2.82843 * x * np.exp(-0.141421 *  np.sqrt(x**2 + y**2)))/np.sqrt(x**2 + y**2) + 3.14159 *  np.sin(2 *  3.1415 *  x) *  np.exp(0.5 *  (np.cos(2 *  3.1415 *  x) + np.cos(2 *  3.1415 *  y))),
        (2.82843 * y * np.exp(-0.141421 *  np.sqrt(x**2 + y**2)))/np.sqrt(x**2 + y**2) + 3.14159 *  np.sin(2 *  3.1415 *  y) *  np.exp(0.5 *  (np.cos(2 *  3.1415 *  x) + np.cos(2 *  3.1415 *  y)))
    ])

def _dd_ackley(X):
    x, y =  X[0], X[1]
    return np.array([
        [
            -(0.4*x**2*np.exp(-0.141421*np.sqrt(x**2 + y**2)))/(x**2 + y**2) - (2.82843*x**2*np.exp(-0.141421*np.sqrt(x**2 + y**2)))/(x**2 + y**2)**(3/2) + (2.82843*np.exp(-0.141421*np.sqrt(x**2 + y**2)))/np.sqrt(x**2 + y**2) + 19.7392*np.cos(2*3.1415*x)*np.exp(0.5*(np.cos(2*3.1415*x) + np.cos(2*3.1415*y))) - 9.8696*(np.sin(2*3.1415*x))**2 *np.exp(0.5*(np.cos(2*3.1415*x) + np.cos(2*3.1415*y))),
            -(0.4 * x * y * np.exp(-0.141421 * np.sqrt(x**2 + y**2)))/(x**2 + y**2) - (2.82843 * x * y*  np.exp(-0.141421 * np.sqrt(x**2 + y**2)))/(x**2 + y**2)**(3/2) - 9.8696 * np.sin(2 * 3.1415 * x) * np.sin(2 * 3.1415 * y) * np.exp(0.5 * (np.cos(2 * 3.1415 * x) + np.cos(2 * 3.1415 * y)))
        ],
        [
            -(0.4 * x * y * np.exp(-0.141421 * np.sqrt(x**2 + y**2)))/(x**2 + y**2) - (2.82843 * x * y * np.exp(-0.141421 * np.sqrt(x**2 + y**2)))/(x**2 + y**2)**(3/2) - 9.8696 * np.sin(2 * 3.1415 * x) * np.sin(2 * 3.1415 * y) * np.exp(0.5 * (np.cos(2 * 3.1415 * x) + np.cos(2 * 3.1415 * y))),
            -(0.4 * y**2 * np.exp(-0.141421 * np.sqrt(x**2 + y**2)))/(x**2 + y**2) - (2.82843 * y**2 * np.exp(-0.141421 * np.sqrt(x**2 + y**2)))/(x**2 + y**2)**(3/2) + (2.82843 * np.exp(-0.141421 * np.sqrt(x**2 + y**2)))/np.sqrt(x**2 + y**2) + 19.7392 * np.cos(2 * 3.1415 * y) * np.exp(0.5 * (np.cos(2 * 3.1415 * x) + np.cos(2 * 3.1415 * y))) - 9.8696 * (np.sin(2 * 3.1415 * y))**2 * np.exp(0.5 * (np.cos(2 * 3.1415 * x) + np.cos(2 * 3.1415 * y)))
        ]
    ])


# ------------- EXPORTS ---------------

f1 = Func(_f1, _df1, _ddf1)

himmelblau = Func(_himmelblau, _d_himmelblau, _dd_himmelblau)
ackley = Func(_ackley, _d_ackley, _dd_ackley)
