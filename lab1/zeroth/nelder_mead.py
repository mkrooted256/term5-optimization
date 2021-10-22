import numpy as np

EXPAND_MINIMIZE, EXPAND_CLASSIC = 0, 1

class NelderMead:    
    #static
    def init_simplex(self, N, x0=None, simplex_edge=1):
        if x0 is None:
            x0 = np.zeros(N)
        S = [x0]
        for i in range(N):
            S.append(x0 + simplex_edge * np.array([int(j == i) for j in range(N)]))
        return np.array(S)
            
    def __init__(self, α=1, β=0.5, γ=2, δ=0.5, expand=EXPAND_MINIMIZE, eps_x=1e-5, eps_f=1e-5, timeout=100):
        """
        EXPAND_MINIMIZE: x_new = argmin{f_e, f_r};
        EXPAND_CLASSIC: x_new = x_e if f_e < f_best
        """
        self.expand = expand
        self.α, self.β, self.γ, self.δ = α, β, γ, δ
        self.eps_x, self.eps_f = eps_x, eps_f
        self.timeout = timeout

    def _eval_vertices(self):
        """Return sorted list of (v_i, f(v_i))"""
        return sorted(enumerate(self.f(self.S[:,0], self.S[:,1])), key=lambda x:x[1])

    def _centroid(self):
        # sum everything except the worst point
        c = self.S.sum(axis=0) - self.S[self.fS[-1][0]]
        return c / self.N  # mean of (N+1)-1 vertices
    
    def _step(self):
        self.steps += 1
        (best_i, best_f), (worst_i, worst_f), (worst2_i, worst2_f) = self.fS[0], self.fS[-1], self.fS[-2]
        c = self._centroid()
        
        action = 'reflected'
        
        # reflect
        x_r = c + self.α * (c - self.S[worst_i])
        f_r = self.f(*x_r)
        
        if f_r < best_f:
            # expand
            x_e = c + self.γ * (x_r - c)
            f_e = self.f(*x_e)
            
            if self.expand == EXPAND_MINIMIZE:
                if f_e < f_r:
                    self.S[worst_i] = x_e
                    action = 'expanded_m'
                else:
                    self.S[worst_i] = x_r
                    action = 'reflected'
            elif self.expand == EXPAND_CLASSIC:
                if f_e < best_f:
                    self.S[worst_i] = x_e
                    action = 'expanded_c'
                else:
                    self.S[worst_i] = x_r
                    action = 'reflected'
        elif f_r > worst2_f:
            # contract
            if f_r < worst_f:
                # contract outside - from reflected to centroid
                x_c = c + self.β * (x_r - c)
                action = 'contracted inside'
            else:
                # contract inside - from worst to centroid 
                x_c = c + self.β * (self.S[worst_i] - c)
                action = 'contracted outside'
            f_c = self.f(*x_c)
            if f_c <= f_r:
                self.S[worst_i] = x_c
            else:
                # shrink
                action = 'shrink'
                for i in range(self.N):
                    x_i = self.fS[i+1][0]
                    self.S[x_i] = self.S[best_i] + self.δ * (self.S[x_i] - self.S[best_i])
        else:
            # best_f <= self.f(xr) < worst2_f
            # enhanced a bit. save new vertex
            self.S[worst_i] = x_r
            action = 'reflected'
        
        self.fS = self._eval_vertices()
        self.c = c
        self.last_action = action
        
    def _term_condition(self):
        if self.steps == 0: 
            return False
        
        # a) vertex are close
        if ((self.S - self.c) < self.eps_x).all():
            self.term_condition = 'x close'
            return True
        
        # b) vertex values are close
        fc = self.f(*self.c)
        if (np.abs([self.f(*x) - fc for x in self.S]) < self.eps_f).all():
            self.term_condition = 'f close'
            return True
        
        # c) timeout
        if self.steps > self.timeout:
            self.term_condition = 'timeout'
            return True
        
        return False
    
    def optimize(self, f, N, G, S0=None, init_only=False):
        self.lims = G.min(axis=0), G.max(axis=0)
        self.N = N
        self.f = f
        self.G = G
        self.steps = 0
        self.last_action = None
        self.term_condition = None
        
        if S0 is None:
            self.S = self.init_simplex(self.N)
        else:
            self.S = S0
        self.fS = self._eval_vertices()
        
        if init_only:
            return
        
        while not self._term_condition():
            self._step()
        
        c = self.S.mean(axis=0)
        return {
            'x': c,
            'f': f(*c),
            'steps': self.steps,
            'message': 'stopped because '+self.term_condition
        }
        