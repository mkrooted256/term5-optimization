import numpy as np
import matplotlib.pyplot as plt

class Dumb:
    def __init__(self, N=100, eps=None):
        self.N, self.eps = N, eps
    
    def optimize(self, f, G):
        """Optimize f:R^1->R
        The Dumb 0th order method. Sample function at N uniformly distributed points and return (x*, f(x*)) with min f(x*)

        If esp is given, then N = domain/eps and provided N is ignored

        return (f(x*), x*)
        """
        lims = G.min(), G.max()
        if self.eps is None:
            X = np.linspace(*lims, self.N)
        else:
            X = np.arange(*lims, self.eps)

        x0 = np.argmin(f(X))
        return X[x0], f(X[x0])
    
    def demo(self, f, G, *options):
        plt.plot(G, f(G))
        x0, fx0 = self.optimize(f, G, *options)
        plt.scatter(x0, fx0, c='r')

        