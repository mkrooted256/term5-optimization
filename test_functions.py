import numpy as np

def _f1(x):
    return -(x*x + (x*x+40*x)*np.sin(x)**3)
def _df1(x):
    return -2*x - (2*x + 40)*np.sin(x)**3 - 3*(x**2 + 40*x)*np.sin(x)**2*np.cos(x)
def _ddf1(x):
    return -6*(2*x + 40)*np.sin(x)**2*np.cos(x) + 3*(x**2 + 40*x)*np.sin(x)**3 - 6*(x**2 + 40*x)*np.sin(x)*np.cos(x)**2 - 2*np.sin(x)**3 - 2


def _f2(x, y):
    return x*x + y*y - 5*x + 8*y

def _himmelblau(x,y):
    return np.square(x*x + y - 11) + np.square(x + y*y -7)

def _ackley(x,y):
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x*x + y*y))) - np.exp(0.5*(np.cos(2*3.1415*x)+np.cos(2*3.1415*y)))+ np.e + 20


# -------

class Func:
    def __init__(self, f, df=None, ddf=None, auto_dx=None):
        self._f, self._d, self._dd = f, df, ddf
        
        if not auto_dx is None :
            self._dx = auto_dx
            self._dxn = np.norm(auto_dx)
            def _auto_d(x0):
                return (self.f(x0 + self._dx) - self.f(x0)) / self._dxn
            self._d = _auto_d
        
    def __call__(self, *args, **kwargs):
        return self._f(*args, **kwargs)

    def d(self, *args, **kwargs):
        return self._d(*args, **kwargs)
    
    def dd(self, *args, **kwargs):
        return self._dd(*args, **kwargs)
    
f1 = Func(_f1, _df1, _ddf1)
f2 = Func(f2)
himmelblau = Func(_himmelblau, auto_dx=1e-6)
ackley = Func(_ackley, auto_dx=1e-6)
    
    