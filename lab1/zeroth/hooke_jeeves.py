import numpy as np

class HookJeeves:                
    def __init__(self, dx=1, div_dx_by=2, eps_x=1e-5, eps_f=1e-5, timeout=100):
        self.eps_x, self.eps_f = eps_x, eps_f
        self.timeout = timeout
        self.dx_0 = dx
        self.div_dx_by = div_dx_by

    def _step(self):
        self.steps += 1

        optimization_failed = True
        new_x = self.last_x
        new_f = self.last_f

        for dim in range(self.ndims):
            delta_x = np.zeros(self.ndims)
            delta_x[dim] = self.dx

            x1 = new_x + delta_x
            f1 = self.f(x1)
            if f1 < new_f:
                optimization_failed = False
                new_x = x1
                new_f = f1
                continue

            x2 = new_x - delta_x
            f2 = self.f(x2)
            if f2 < new_f:
                optimization_failed = False
                new_x = x2
                new_f = f2
        
        if optimization_failed:
            self.dx /= self.div_dx_by
            return None, None

        return new_x, new_f
        
    def _term_condition(self):
        if self.steps > self.timeout:
            self.message = 'timeout'
            return True

        # print(self.x_history)

        if len(self.x_history) < 2: 
            return False
        
        if np.linalg.norm(self.x_history[-1]-self.x_history[-2]) < self.eps_x:
            self.message = 'x convergence'
            return True

        if np.abs(self.f_history[-1]-self.f_history[-2]) < self.eps_f:
            self.message = 'f convergence'
            return True
        
        return False
    
    def optimize(self, f, x_start=(0,0), ndims=2):
        self.f = f
        self.x_history = []
        self.f_history = []
        self.steps = 0
        self.dx = self.dx_0
        self.ndims = ndims

        self.last_x = np.array(x_start)
        self.last_f = f(self.last_x)
        
        self.x_history.append(self.last_x)
        self.f_history.append(self.last_f)

        while not self._term_condition():
            new_x, new_f = self._step()

            if new_x is not None:
                self.x_history.append(new_x)
                self.f_history.append(new_f)

                self.last_x, self.last_f = new_x, new_f
        
        return {
            'x': self.last_x,
            'f': self.last_f,
            'x_history': np.array(self.x_history),
            'f_history': self.f_history,
            'steps': self.steps,
            'message': 'stopped because ' + self.message
        }
        