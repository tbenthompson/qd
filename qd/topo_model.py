import numpy as np
from scipy.sparse.linalg import cg
import scipy.sparse

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

def check_naninf(x):
    return np.any(np.isnan(x)) or np.any(np.isinf(x))

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
            if check_naninf(y):
                return np.inf * y
            data = self.solve_for_full_state(t, y)
            if not data:
                return np.inf * y
            disp_slip, state, traction, fault_V, dstatedt, surf_vel = data
            # return np.concatenate((np.zeros(self.m.n_dofs('surf')), fault_V, dstatedt))
            return np.concatenate((fault_V, dstatedt, surf_vel))
        return derivs

    def get_components(self, y):
        n_fault_tris = self.m.n_tris('fault')
        slip_end = n_fault_tris * 9
        state_end = slip_end + n_fault_tris * 3
        slip = y[0:slip_end]
        state = y[slip_end:state_end]
        disp = y[state_end:]
        return slip, state, disp

    def solve_for_full_state(self, t, y):
        timer = self.cfg['Timer']()
        do_fast_step = self.last_vel is not None and self.cfg['fast_surf_disp']

        slip, state, surf_disp = self.get_components(y)

        if np.any(state < 0) or np.any(state > 1.2):
            print(state)

        plate_dist = t * self.cfg['plate_rate']
        plate_motion = (plate_dist * self.field_inslipdir).reshape(-1)
        slip_deficit = self.ones_interior * (plate_motion - slip)
        timer.report('slip deficit')

        if do_fast_step:
            disp_slip = np.concatenate((surf_disp, slip))
        else:
            disp_slip = self.slip_to_disp(slip_deficit)
        timer.report(f'disp_slip(fast={do_fast_step})')

            # V = (disp - old_disp) /
        # disp_slip = np.concatenate(disp, slip_deficit)
        traction = self.disp_slip_to_traction(disp_slip)
        fault_traction = traction[self.m.n_dofs('surf'):].copy()
        timer.report('traction')

        fault_V = rate_state_solve(self, fault_traction, state)
        timer.report('fault_V')

        dstatedt = state_evolution(self.cfg, fault_V, state)
        timer.report('dstatedt')

        if check_naninf(fault_V):
            return False

        if do_fast_step:
            surf_vel = self.m.get_dofs(
                self.slip_rate_to_surf_velocity(self.last_vel, fault_V),
                'surf'
            )
        else:
            surf_vel = np.zeros(self.m.n_dofs('surf'))
        timer.report('surf_vel')

        out = disp_slip, state, traction, fault_V, dstatedt, surf_vel
        return out

    def post_step(self, ts, ys):
        if len(ts) >= 2:
            dt = ts[-1] - ts[-2]
            dy = ys[-1] - ys[-2]
            dydt = dy / dt
            self.last_vel = dydt[self.m.n_tris('fault') * 12:]

    def display_model(self, t, y, plotter = plot_fields):
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
        plotter(self, surf_vel, which = 'surf', dims = [0,1])
        print('traction on fault')
        plotter(self, self.m.get_dofs(traction, 'fault'))
        print('traction on surface')
        plotter(self, self.m.get_dofs(traction, 'surf'), which = 'surf', dims = [0,1])
        print('state')
        plotter(self, state)

    @property
    @remember
    def slip_to_disp(self):
        return get_slip_to_disp(self.m, self.cfg, self.T())

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

    @remember
    def T(self):
        setup_logging(self.cfg)
        return build_elastic_op(self.m, self.cfg, 'T')

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
    n_surf = m.n_dofs('surf')
    n_fault = m.n_dofs('fault')
    n_total = m.n_dofs()

    cs = tct.continuity_constraints(m.pts, m.tris, m.get_start('fault'))
    cs.extend(tct.free_edge_constraints(m.get_tris('surf')))
    cm, c_rhs = tct.build_constraint_matrix(cs, m.n_dofs())

    T = build_elastic_op(
        m, cfg, 'T',
        obs_subset = m.get_tri_idxs('surf'),
        src_subset = np.arange(m.tris.shape[0])
    )

    disp_mass_op = tct.MassOp(
        cfg['tectosaur_cfg']['quad_mass_order'], m.pts, m.tris
    ).mat * 0.5
    lil_mat = disp_mass_op.tolil()
    lil_mat[n_surf:, n_surf:] = scipy.sparse.identity(n_fault)
    disp_mass_op = lil_mat.tocsr()
    constrained_mass_op = cm.T.dot(disp_mass_op.dot(cm))

    def f(in_surf_vel, in_fault_vel):
        t = cfg['Timer']()
        full_vel = np.concatenate((in_surf_vel, in_fault_vel))
        t.report('concat')
        rhs = np.empty(full_vel.shape[0])
        rhs[:m.n_dofs('surf')] = -T.dot(full_vel)
        t.report('Tdot')
        rhs[m.n_dofs('surf'):] = in_fault_vel
        t.report('set rhs fault')
        constrained_rhs = cm.T.dot(rhs)
        t.report('constrain rhs')
        out = cm.dot(cg(constrained_mass_op, constrained_rhs)[0])
        t.report('solve')
        return out
    return f


def get_slip_to_disp(m, cfg, T):
    base_cs = tct.continuity_constraints(m.pts, m.tris, m.get_start('fault'))
    base_cs.extend(tct.free_edge_constraints(m.get_tris('surf')))

    mass_op = tct.MassOp(
        cfg['tectosaur_cfg']['quad_mass_order'], m.pts, m.tris
    ).mat

    def f(slip):
        cs = base_cs + tct.all_bc_constraints(
            m.n_tris('surf'), m.n_tris(), slip
        )
        cm, c_rhs = tct.build_constraint_matrix(cs, m.n_dofs())

        iop = tct.SumOp([T, tct.MultOp(mass_op, 0.5)])
        rhs = -iop.dot(c_rhs)
        out = tectosaur_topo.solve.iterative_solve(
            iop, cm, rhs, lambda x: x, dict(solver_tol = 1e-6)
        )
        return out + c_rhs
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

        rhs = H.dot(disp_slip)
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
    f.H = H
    f.cm = cm
    f.traction_mass_op = traction_mass_op
    return f
