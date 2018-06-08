import numpy as np

from scipy.integrate import RK45

from .derivs import make_derivs

siay = 60 * 60 * 24 * 365.25

class Integrator:
    def __init__(self, model, init_conditions, save_freq = 100):
        self.model = model
        self.derivs = make_derivs(model)
        self.init_t, self.init_slip, self.init_state = init_conditions
        self.init_conditions = np.concatenate((self.init_slip, self.init_state))
        self.rk45 = RK45(
            self.derivs,
            self.init_t,
            self.init_conditions,
            1e50,
            atol = self.model.cfg['timestep_tol'],
            rtol = self.model.cfg['timestep_tol']
        )
        self.h_t = []
        self.h_y = []

    def integrate(self, n_steps, until = None, display_fnc = None):
        for i in range(n_steps):
            if until is not None and integrator.t > until:
                return
            assert(self.rk45.step() == None)
            self.h_t.append(self.rk45.t)
            self.h_y.append(self.rk45.y.copy())

            if display_fnc is not None:
                display_fnc(self)

            if len(self.h_t) %


            # clear_output()
            # print(i, integrator.t / siay)
            # plot_setting(integrator.t, integrator.y, qdm, self.cfg, slip_to_traction)
