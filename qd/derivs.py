import numpy as np
from scipy.optimize import fsolve

import cppimport.import_hook
from .pt_average import pt_averageD
from . import newton

def make_derivs(model):
    def derivs(t, y):
        slip, slip_deficit, state, traction, V, dstatedt = solve_for_full_state(
            model, t, y
        )
        return np.concatenate((V, dstatedt))
    return derivs

def separate_slip_state(y):
    n_total_dofs = y.shape[0]
    n_slip_dofs = n_total_dofs // 4 * 3
    return y[:n_slip_dofs], y[n_slip_dofs:]

def get_plate_motion(model, t):
    dist = t * model.cfg['plate_rate']
    return dist * model.field_inslipdir

def get_slip_deficit(model, t, slip):
    out = model.ones_interior * (get_plate_motion(model, t).reshape(-1) - slip)
    return out

def pt_average(pts, tris, field):
    avg_field = np.empty_like(field)
    for i in range(pts.shape[0]):
        tri_idxs, corner_idxs = np.where(tris == i)
        idxs = tri_idxs * 3 + corner_idxs
        val = np.mean(field[idxs])
        avg_field[idxs] = val
    return avg_field

def rate_state_solve(model, traction, state):
    timer = model.cfg['Timer']()
    V = np.empty_like(model.field_inslipdir)
    newton.rate_state_solver(
        model.tri_normals, traction, state, V,
        model.cfg['a'], model.cfg['eta'], model.cfg['V0'],
        model.cfg['additional_normal_stress'],
        1e-12, 50, int(model.n_dofs / model.n_tris),
        model.cfg.get('rs_separate_dims', False)
    )
    timer.report('newton')

    if model.basis_dim == 1:
        ptavg_V = V
    else:
        ptavg_V = np.empty_like(V)
        for d in range(3):
            ptavg_V.reshape((-1,3))[:,d] = pt_averageD(
                model.m.pts, model.m.tris, V.reshape((-1,3))[:,d].copy()
            )
    timer.report('pt avg')

    out = (
        model.field_inslipdir_edges * model.cfg['plate_rate']
        + model.ones_interior * ptavg_V
    )
    timer.report('rate_state_solve --> out')
    return out

# State evolution law -- aging law.
def aging_law(cfg, V, state):
    return (cfg['b'] * cfg['V0'] / cfg['Dc']) * (
        np.exp((cfg['f0'] - state) / cfg['b']) - (V / cfg['V0'])
    )

def state_evolution(cfg, V, state):
    V_mag = np.linalg.norm(V.reshape(-1,3), axis = 1)
    return aging_law(cfg, V_mag, state)

def solve_for_full_state(model, t, y):
    timer = model.cfg['Timer']()
    slip, state = separate_slip_state(y)
    timer.report('separate_slip_state')
    slip_deficit = get_slip_deficit(model, t, slip)
    timer.report('get_slip_deficit')
    traction = model.slip_to_traction(slip_deficit)
    timer.report('slip_to_traction')
    V = rate_state_solve(model, traction, state)
    timer.report('rate_state_solve')
    dstatedt = state_evolution(model.cfg, V, state)
    timer.report('state_evolution')
    return slip, slip_deficit, state, traction, V, dstatedt

def init_zero_slip(model):
    t = 0
    init_slip = np.zeros((model.m.tris.shape[0] * 9))
    init_state = np.ones((model.m.tris.shape[0] * 3))
    return t, init_slip, init_state

def init_creep(model, traction_to_slip):
    V_i = model.cfg['plate_rate']
    def f(state):
        return aging_law(model.cfg, V_i, state)
    state_i = fsolve(f, 0.7)[0]
    sigma_n = model.cfg['additional_normal_stress']
    tau_i = newton.F(V_i, sigma_n, state_i, model.cfg['a'][0], model.cfg['V0'])
    init_traction = tau_i * model.field_inslipdir_interior
    init_slip_deficit = traction_to_slip(init_traction)
    init_state =  state_i * np.ones((model.n_dofs))
    return 0, -init_slip_deficit, init_state
