import numpy as np

def _f1(x):
    return -(x*x + (x*x+40*x)*np.sin(x)**3)
def _df1(x):
    return -2*x - (2*x + 40)*np.sin(x)**3 - 3*(x**2 + 40*x)*np.sin(x)**2*np.cos(x)
def _ddf1(x):
    return -6*(2*x + 40)*np.sin(x)**2*np.cos(x) + 3*(x**2 + 40*x)*np.sin(x)**3 - 6*(x**2 + 40*x)*np.sin(x)*np.cos(x)**2 - 2*np.sin(x)**3 - 2

# -------

class Func:
    def __init__(self, f, df=None, ddf=None):
        self.f, self.d, self.dd = f, df, ddf
        
    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

f1 = Func(_f1, _df1, _ddf1)
    
    