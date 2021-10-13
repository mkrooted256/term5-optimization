import numpy as np

class DumbGradientDescent:
    def __init__(self, x_eps=1e-10, f_eps=1e-10, timeout = 100, rate=1):
        self.x_eps = x_eps
        self.f_eps = f_eps
        self.timeout = timeout
        self.rate = rate
    
    def __str__(self):
        return f"DumbGradientDescent (x_eps={self.x_eps}, f_eps={self.f_eps}, timeout={self.timeout}, rate={self.rate})"

    def optimize(self, f, df, x_start=None, ndims=2):
        if x_start is None:
            x_start = np.zeros(ndims)
        self.ndims = ndims
        self.f = f
        self.x_history = []
        self.f_history = []
        self.steps = 0

        new_x = last_x = x_start
        new_f = last_f = f(x_start)
        self.x_history.append(new_x)
        self.f_history.append(new_f)
        while True:
            last_x = new_x
            last_f = new_f

            new_x = last_x - df(last_x)*self.rate
            
            self.x_history.append(new_x)
            self.f_history.append(new_f)

            if (np.linalg.norm(new_x - last_x) < self.x_eps):
                msg = 'stopped by x convergence'
                break
            if (new_f - last_f > self.f_eps):
                msg = 'stopped by f(x) convergence'
                break
            if (self.steps >= self.timeout):
                msg = 'stopped by timeout'
                break
            self.steps+=1

        return {
            'x': new_x,
            'f': new_f,
            'steps': self.steps,
            'x_history': np.array(self.x_history),
            'f_history': self.f_history,
            'message': msg
        }
        
