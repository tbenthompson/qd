import numpy as np
from scipy.sparse.linalg import cg
from IPython.display import clear_output

import tectosaur as tct
import tectosaur_topo
from tectosaur.util.geometry import unscaled_normals
from tectosaur.util.timer import Timer

from . import siay
from .full_model import setup_logging
from .model_helpers import (
    calc_derived_constants, remember, rate_state_solve,
    state_evolution, build_elastic_op)
from .plotting import plot_fields

class TopoModel:
    def __init__(self, m, cfg):
        cfg = calc_derived_constants(cfg)
        self.cfg = cfg
        self.cfg['Timer'] = self.cfg.get(
            'Timer',
            lambda: Timer(output_fnc = lambda x: None)
        )
        self.setup_mesh(m)
        self.setup_edge_bcs()
        self.last_vel = None

    def make_derivs(self):
        def derivs(t, y):
            if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                return np.inf * y
            data = self.solve_for_full_state(t, y)
            disp_slip, state, traction, fault_V, dstatedt, surf_vel = data
            # return np.concatenate((np.zeros(self.m.n_dofs('surf')), fault_V, dstatedt))
            return np.concatenate((fault_V, dstatedt, surf_vel))
        return derivs

    def solve_for_full_state(self, t, y):
        do_fast_step = self.last_vel is not None and self.cfg['fast_surf_disp']

        n_fault_tris = self.m.n_tris('fault')
        slip_end = n_fault_tris * 9
        state_end = slip_end + n_fault_tris * 3
        slip = y[0:slip_end]
        state = y[slip_end:state_end]

        if np.any(state < 0) or np.any(state > 1.2):
            print(state)

        plate_dist = t * self.cfg['plate_rate']
        plate_motion = (plate_dist * self.field_inslipdir).reshape(-1)
        slip_deficit = self.ones_interior * (plate_motion - slip)

        print('solve for disp')
        if do_fast_step:
            print('FASTSTEP')
            surf_disp = y[state_end:]
            disp_slip = np.concatenate((surf_disp, slip))
        else:
            disp_slip = self.slip_to_disp(slip_deficit)

            # V = (disp - old_disp) /
        # disp_slip = np.concatenate(disp, slip_deficit)
        print('solve for traction')
        traction = self.disp_slip_to_traction(disp_slip)
        fault_traction = traction[self.m.n_dofs('surf'):].copy()
        print('solve for V')
        fault_V = rate_state_solve(self, fault_traction, state)
        print('solve for dstatedt')
        dstatedt = state_evolution(self.cfg, fault_V, state)

        if do_fast_step:
            surf_vel = self.slip_rate_to_surf_velocity(fault_V, self.last_vel)
        else:
            surf_vel = np.zeros(self.m.n_dofs('surf'))

        out = disp_slip, state, traction, fault_V, dstatedt, surf_vel
        return out

    def post_step(self, ts, ys):
        if len(ts) >= 2:
            dt = ts[-1] - ts[-2]
            dy = ys[-1] - ys[-2]
            dydt = dy / dt
            self.last_vel = dydt[self.m.n_tris('fault') * 12:]

    def display_model(self, t, y, plotter = plot_fields):
        clear_output(wait = True)
        print(t / siay)
        data = self.solve_for_full_state(t, y)
        disp_slip, state, traction, fault_V, dstatedt, surf_vel = data
        print('slip')
        plotter(self, self.m.get_dofs(disp_slip, 'fault'))
        print('surface displacement')
        plotter(self, self.m.get_dofs(disp_slip, 'surf'), which = 'surf', dims = [0,1])
        print('fault V')
        plotter(self, np.log10(np.abs(fault_V) + 1e-40))
        print('surf V')
        plotter(self, np.log10(np.abs(surf_vel) + 1e-40), which = 'surf', dims = [0,1])
        print('traction on fault')
        plotter(self, self.m.get_dofs(traction, 'fault'))
        print('traction on surface')
        plotter(self, self.m.get_dofs(traction, 'surf'), which = 'surf', dims = [0,1])
        print('state')
        plotter(self, state)

    @property
    @remember
    def slip_to_disp(self):
        return get_slip_to_disp(self.m, self.cfg, self.H())

    @property
    @remember
    def slip_rate_to_surf_velocity(self):
        return get_slip_rate_to_surf_velocity(self.m, self.cfg)

    @property
    @remember
    def disp_slip_to_traction(self):
        return get_disp_slip_to_traction(self.m, self.cfg, self.H())

    @property
    @remember
    def full_traction_to_slip(self):
        return get_traction_to_slip(self.m, self.cfg, self.H())

    @property
    def traction_to_slip(self):
        def f(traction):
            full_traction = np.zeros(self.m.n_dofs())
            full_traction[self.m.n_dofs('surf'):] = traction
            return self.full_traction_to_slip(full_traction)
        return f

    @remember
    def H(self):
        setup_logging(self.cfg)
        return build_elastic_op(self.m, self.cfg, 'H')

    def setup_mesh(self, m):
        self.m = m
        self.fault_start_idx = m.get_start('fault')
        fault_tris = self.m.get_tris('fault')

        self.unscaled_tri_normals = unscaled_normals(self.m.pts[fault_tris])
        self.tri_size = np.linalg.norm(self.unscaled_tri_normals, axis = 1)
        self.tri_normals = self.unscaled_tri_normals / self.tri_size[:, np.newaxis]

        self.n_tris = self.m.tris.shape[0]
        self.basis_dim = 3
        self.n_dofs = self.basis_dim * self.n_tris

    def setup_edge_bcs(self):
        cs = tct.free_edge_constraints(self.m.tris)
        cs = [c for c in cs if c.terms[0].dof >= self.m.n_dofs('surf')]
        cm, c_rhs = tct.build_constraint_matrix(cs, self.m.n_dofs())
        constrained_slip = np.ones(cm.shape[1])
        self.ones_interior = cm.dot(constrained_slip)[self.m.n_dofs('surf'):]

        self.field_inslipdir_interior = self.ones_interior.copy()
        self.field_inslipdir = self.ones_interior.copy()
        for d in range(3):
            val = self.cfg['slipdir'][d]
            self.field_inslipdir_interior.reshape((-1,3))[:,d] *= val
            self.field_inslipdir.reshape((-1,3))[:,d] = val

        self.field_inslipdir_edges = self.field_inslipdir - self.field_inslipdir_interior

def get_slip_rate_to_surf_velocity(m, cfg):
    base_cs = tct.continuity_constraints(m.pts, m.tris, m.get_start('fault'))
    base_cs.extend(tct.free_edge_constraints(m.get_tris('surf')))

    T = build_elastic_op(
        m, cfg, 'T',
        # obs_subset = m.get_tri_idxs('surf'),
        # src_subset = np.arange(m.n_tris())
    )
    disp_mass_op = tct.MassOp(
        cfg['tectosaur_cfg']['quad_mass_order'], m.pts, m.tris
    ).mat * 0.5

    def f(in_surf_vel, in_fault_vel):
        cs = base_cs + tct.all_bc_constraints(
            m.n_tris('surf'), m.n_tris(), in_fault_vel
        )
        cm, c_rhs = tct.build_constraint_matrix(cs, m.n_dofs())
        constrained_disp_mass_op = cm.T.dot(disp_mass_op.dot(cm))

        full_vel = np.concatenate((in_surf_vel, in_fault_vel))
        rhs = disp_mass_op.dot(c_rhs)
        rhs += T.dot(full_vel)
        constrained_rhs = cm.T.dot(rhs)
        soln = cg(constrained_disp_mass_op, constrained_rhs)#, callback = callback)
        return cm.dot(soln[0]) + c_rhs
    return f


def get_slip_to_disp(m, cfg, H):
    base_cs = tct.continuity_constraints(m.pts, m.tris, m.get_start('fault'))
    base_cs.extend(tct.free_edge_constraints(m.get_tris('surf')))
    def f(slip):
        cs = base_cs
        if slip is not None:
            cs = base_cs + tct.all_bc_constraints(
                m.n_tris('surf'), m.n_tris(), slip
            )
        cm, c_rhs = tct.build_constraint_matrix(cs, m.n_dofs())
        rhs = -H.dot(c_rhs)
        out = tectosaur_topo.solve.iterative_solve(
            H, cm, rhs, lambda x: x, dict(solver_tol = 1e-6)
        )
        return out + c_rhs
        # import matplotlib.pyplot as plt
        # plt.plot(c_rhs)
        # plt.show()
    return f

def get_disp_slip_to_traction(m, cfg, H):
    csS = tct.continuity_constraints(m.pts, m.get_tris('surf'), m.get_end('surf'))
    csS.extend(tct.free_edge_constraints(m.get_tris('surf')))
    csF = tct.continuity_constraints(m.pts, m.get_tris('fault'), m.get_end('fault'))
    csF.extend(tct.free_edge_constraints(m.get_tris('fault')))
    cs = tct.build_composite_constraints((csS, 0), (csF, m.n_dofs('surf')))
    cm, c_rhs = tct.build_constraint_matrix(cs, m.n_dofs())

    traction_mass_op = tct.MassOp(cfg['tectosaur_cfg']['quad_mass_order'], m.pts, m.tris)
    constrained_traction_mass_op = cm.T.dot(traction_mass_op.mat.dot(cm))

    def f(disp_slip):
        def callback(x):
            callback.iter += 1
            print(callback.iter)
        callback.iter = 0

        rhs = -H.dot(disp_slip)
        constrained_rhs = cm.T.dot(rhs)
        soln = cg(constrained_traction_mass_op, constrained_rhs)#, callback = callback)
        out = cfg['sm'] * cm.dot(soln[0])
        return out
    return f

def get_traction_to_slip(m, cfg, H):
    csS = tct.continuity_constraints(m.pts, m.tris, m.get_start('fault'))
    csF = tct.continuity_constraints(m.pts, m.get_tris('fault'), m.get_end('fault'))
    cs = tct.build_composite_constraints((csS, 0), (csF, m.n_dofs('surf')))
    cs.extend(tct.free_edge_constraints(m.tris))

    cm, c_rhs = tct.build_constraint_matrix(cs, m.n_dofs())

    traction_mass_op = tct.MassOp(cfg['tectosaur_cfg']['quad_mass_order'], m.pts, m.tris)
    np.testing.assert_almost_equal(c_rhs, 0.0)

    def f(traction):
        rhs = -traction_mass_op.dot(traction / cfg['sm'])
        out = tectosaur_topo.solve.iterative_solve(
            H, cm, rhs, lambda x: x, dict(solver_tol = 1e-6)
        )
        return out
    return f
