import numpy as np
from scipy.sparse.linalg import cg

import tectosaur as tct
import tectosaur_topo
from tectosaur.util.geometry import unscaled_normals
from tectosaur.util.timer import Timer

from .full_model import build_hypersingular, setup_logging
from .derivs import rate_state_solve, state_evolution
from .model_helpers import calc_derived_constants, remember

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

    def make_derivs(self):
        def derivs(t, y):
            disp_end = self.m.n_dofs('surf')
            slip_end = disp_end + self.m.n_dofs('fault')
            old_disp = y[:disp_end]
            slip = y[disp_end:slip_end]
            state = y[slip_end:]

            plate_dist = t * self.cfg['plate_rate']
            plate_motion = (plate_dist * self.field_inslipdir).reshape(-1)
            slip_deficit = self.ones_interior * (plate_motion - slip)

            solve_for_disp = True
            if solve_for_disp:
                disp_slip = self.slip_to_disp(slip_deficit)

                # V = (disp - old_disp) /
            # disp_slip = np.concatenate(disp, slip_deficit)
            traction = self.disp_slip_to_traction(disp_slip)
            fault_V = rate_state_solve(self, traction[self.m.n_dofs('surf'):].copy(), state)
            dstatedt = state_evolution(self.cfg, fault_V, state)
            # slip, slip_deficit, state, traction, V, dstatedt = solve_for_full_state(
            #     self, t, y
            # )

            from IPython.core.debugger import Tracer
            Tracer()()
            out = np.concatenate((np.zeros_like(old_disp), fault_V, dstatedt))
            # if np.any(np.isnan(out)) or np.any(np.isinf(out)):
            return out
        return derivs

    @property
    @remember
    def slip_to_disp(self):
        return get_slip_to_disp(self.m, self.cfg, self.H())

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
        return build_hypersingular(self.m, self.cfg)

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


def get_slip_to_disp(m, cfg, H):
    base_cs = surface_continuity(m, True)
    base_cs.extend(tct.free_edge_constraints(m.tris))
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
    cm, c_rhs = surf_fault_continuity(m, False)
    traction_mass_op = tct.MassOp(cfg['tectosaur_cfg']['quad_mass_order'], m.pts, m.tris)
    constrained_traction_mass_op = cm.T.dot(traction_mass_op.mat.dot(cm))

    def f(disp_slip):
        def callback(x):
            callback.iter += 1
            print(callback.iter)
        callback.iter = 0

        rhs = -H.dot(disp_slip)
        constrained_rhs = cm.T.dot(rhs)
        soln = cg(constrained_traction_mass_op, constrained_rhs, callback = callback)
        out = cfg['sm'] * cm.dot(soln[0])
        return out
    return f

def get_traction_to_slip(m, cfg, H):
    traction_mass_op = tct.MassOp(cfg['tectosaur_cfg']['quad_mass_order'], m.pts, m.tris)
    cm, c_rhs = surf_fault_continuity(m, True)
    np.testing.assert_almost_equal(c_rhs, 0.0)

    def f(traction):
        rhs = -traction_mass_op.dot(traction / cfg['sm'])
        out = tectosaur_topo.solve.iterative_solve(
            H, cm, rhs, lambda x: x, dict(solver_tol = 1e-6)
        )
        return out
    return f

def surf_fault_continuity(m, fault_discontinuity):
    csS = surface_continuity(m, fault_discontinuity)
    csF = fault_continuity(m)

    cs = tct.build_composite_constraints((csS, 0), (csF, m.n_dofs('surf')))
    cs.extend(tct.free_edge_constraints(m.tris))
    return tct.build_constraint_matrix(cs, m.n_dofs())

def surface_continuity(m, fault_discontinuity = True):
    if fault_discontinuity:
        cs = tct.continuity_constraints(m.pts, m.tris, m.get_start('fault'))
    else:
        cs = tct.continuity_constraints(m.pts, m.get_tris('surf'), m.get_end('surf'))
    return cs

def fault_continuity(m):
    cs = tct.continuity_constraints(m.pts, m.get_tris('fault'), m.get_end('fault'))
    return cs
