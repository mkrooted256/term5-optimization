import numpy as np

class NewtonDescent:
    def __init__(self, x_eps=1e-10, f_eps=1e-10, timeout = 100, rate=1, div_step_by=2, monotonous=True):
        self.x_eps = x_eps
        self.f_eps = f_eps
        self.timeout = timeout
        self.rate = float(rate)
        self.div_step_by = div_step_by
        self.monotonous = monotonous
    
    def __str__(self):
        s = f"NewtonDescent (x_eps={self.x_eps}, f_eps={self.f_eps}, N={self.timeout}, r0={self.rate}, rdb={self.div_step_by})"
        if not self.monotonous:
            s = "НЕМОНОТОННИЙ " + s
        return s

    def optimize(self, f, df, ddf, x_start=None, ndims=2):
        if x_start is None:
            x_start = np.zeros(ndims)
        self.ndims = ndims
        self.f = f
        self.x_history = []
        self.f_history = []
        self.steps = 0

        rate = self.rate

        new_x = x_start
        new_f = f(x_start)
        self.x_history.append(new_x)
        self.f_history.append(new_f)
        while True:
            last_x = new_x
            last_f = new_f

            hess = np.linalg.inv(ddf(last_x)) 
            a = hess @ df(last_x) 
            x1 = last_x - a * rate
            f1 = f(x1)

            if f1 > last_f:
                x1 = last_x - a * rate / np.linalg.norm(hess)
                f1 = f(x1)

            if self.monotonous:
                while f1 > last_f:
                    rate /= self.div_step_by
                    x1 = last_x - a * rate / np.linalg.norm(hess)
                    f1 = f(x1)
            
            new_x = x1
            new_f = f1
            
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

        # print("end rate: ", self.rate   )
        return {
            'x': new_x,
            'f': new_f,
            'steps': self.steps,
            'x_history': np.array(self.x_history),
            'f_history': self.f_history,
            'message': msg
        }
        
