import numpy as np

from scipy.integrate import RK45

from . import siay
from .data import ChunkedDataSaver

class Integrator:
    def __init__(
            self, model, init_conditions, data_handler = None,
            init_step_size = siay / 10.0):

        if data_handler is None:
            data_handler = ChunkedDataSaver()

        self.model = model
        self.derivs = model.make_derivs()
        self.init_conditions = init_conditions
        self.h_t = []
        self.h_y = []

        self.data_handler = data_handler
        self.data_handler.initialized(self)

        self.setup_rk45(init_step_size)

    def setup_rk45(self, init_step_size):
        init_t, init_y = self.init_conditions
        self.rk45 = RK45(
            self.derivs,
            init_t,
            init_y,
            1e50,
            atol = self.model.cfg['timestep_tol'],
            rtol = self.model.cfg['timestep_tol']
        )
        self.rk45.h_abs = init_step_size

    def step_idx(self):
        return len(self.h_t)

    def integrate(self, n_steps, until = None, display_fnc = None, display_interval = 1):
        if display_fnc is None:
            display_fnc = lambda: print(self.h_t[-1])

        for i in range(n_steps):
            if until is not None and integrator.t > until:
                return
            assert(self.rk45.step() == None)
            new_t = self.rk45.t
            new_y = self.rk45.y.copy()
            self.h_t.append(new_t)
            self.h_y.append(new_y)
            self.model.post_step(self.h_t, self.h_y)

            if i % display_interval == 0:
                display_fnc(self)
            self.data_handler.stepped(self)


