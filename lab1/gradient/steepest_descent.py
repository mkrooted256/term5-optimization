import numpy as np

class SteepestGradientDescent:
    def __init__(self, x_eps=1e-10, f_eps=1e-10, timeout = 100, rate=1, scalar_opt_steps=100):
        self.x_eps = x_eps
        self.f_eps = f_eps
        self.timeout = timeout
        self.rate = float(rate)
        self.scalar_opt_steps = scalar_opt_steps
    
    def __str__(self):
        return f"SteepestDescent (x_eps={self.x_eps}, f_eps={self.f_eps}, N={self.timeout}, r0={self.rate}, sc_steps={self.scalar_opt_steps})"

    def optimize(self, f, df, x_start=None, ndims=2):
        if x_start is None:
            x_start = np.zeros(ndims)
        self.ndims = ndims
        self.f = f
        self.x_history = []
        self.f_history = []
        self.steps = 0

        rate = self.rate

        new_x = np.array(x_start)
        new_f = f(x_start)
        self.x_history.append(new_x)
        self.f_history.append(new_f)
        while True:
            last_x = new_x
            last_f = new_f

            antigrad = -df(last_x)

            # find best x1 on line "last_x + [0,1] * antigrad"
            all_xs = last_x.reshape(-1,1) + np.linspace(0,1,self.scalar_opt_steps) * rate * antigrad.reshape(-1,1)
            f_vals = f(all_xs)
            
            I = f_vals.argmin()

            new_x = all_xs[:,I]
            new_f = f_vals[I]
            
            self.x_history.append(new_x)
            self.f_history.append(new_f)

            delta_x = np.linalg.norm(new_x - last_x)
            if (delta_x < self.x_eps):
                msg = f'stopped by x convergence. delta x = {delta_x}'
                break
            if (np.abs(new_f - last_f) < self.f_eps):
                msg = f'stopped by f(x) convergence. delta f = {new_f - last_f}'
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
        
