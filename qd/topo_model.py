import numpy as np
from scipy.sparse.linalg import cg
import scipy.sparse

import tectosaur as tct
import tectosaur_topo
from tectosaur.util.geometry import unscaled_normals
from tectosaur.util.timer import Timer
from tectosaur.constraints import ConstraintEQ, Term

from . import siay
from .full_model import setup_logging
from .model_helpers import (
    calc_derived_constants, remember, rate_state_solve,
    state_evolution, build_elastic_op)
from .plotting import plot_fields

def check_naninf(x):
    return np.any(np.isnan(x)) or np.any(np.isinf(x))

class TopoModel:
    def __init__(self, m, cfg):#, which_side):
        cfg = calc_derived_constants(cfg)
        # self.which_side = which_side
        self.cfg = cfg
        self.cfg['Timer'] = self.cfg.get(
            'Timer',
            lambda: Timer(output_fnc = lambda x: None)
        )
        self.setup_mesh(m)
        self.setup_edge_bcs()
        self.last_vel = None
        self.first = True

    def make_derivs(self):
        def derivs(t, y):
            if check_naninf(y):
                return np.inf * y
            data = self.solve_for_full_state(t, y)
            if not data:
                return np.inf * y
            disp_slip, state, traction, fault_V, dstatedt, surf_vel = data
            if self.first:
                self.last_vel = surf_vel
                self.first = False
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

    def do_fast_step(self):
        return self.last_vel is not None and self.cfg['fast_surf_disp']

    def solve_for_disp_slip_deficit(self, t, y):
        slip, state, surf_disp = self.get_components(y)

        if np.any(state < 0) or np.any(state > 1.2):
            print(state)

        plate_dist = t * self.cfg['plate_rate']
        plate_motion = (plate_dist * self.field_inslipdir).reshape(-1)
        slip_deficit = self.ones_interior * (plate_motion - slip)
        surf_disp_deficit = -((self.locked_fault_surf_disp_deficit * plate_dist) + surf_disp)

        if self.do_fast_step():
            disp_slip_deficit = np.concatenate((surf_disp_deficit, slip_deficit))
        else:
            disp_slip_deficit = self.slip_to_disp(slip_deficit)
        return disp_slip_deficit

    def solve_for_full_state(self, t, y):
        timer = self.cfg['Timer']()
        _, state, _ = self.get_components(y)
        disp_slip_deficit = self.solve_for_disp_slip_deficit(t, y)
        do_fast_step = self.do_fast_step()
        timer.report(f'disp_slip(fast={do_fast_step})')

        traction = self.disp_slip_to_traction(disp_slip_deficit)
        fault_traction = traction[self.m.n_dofs('surf'):].copy()
        timer.report('traction')

        fault_V = rate_state_solve(self, fault_traction, state)
        timer.report('fault_V')

        dstatedt = state_evolution(self.cfg, fault_V, state)
        timer.report('dstatedt')

        if check_naninf(fault_V):
            return False

        if self.do_fast_step():
            surf_vel = self.m.get_dofs(
                self.slip_rate_to_surf_velocity(self.last_vel, fault_V),
                'surf'
            )
        else:
            surf_vel = np.zeros(self.m.n_dofs('surf'))
        timer.report('surf_vel')

        out = disp_slip_deficit, state, traction, fault_V, dstatedt, surf_vel
        return out

    def post_step(self, ts, ys, rk):
        if len(ts) >= 2:
            dt = ts[-1] - ts[-2]
            dy = ys[-1] - ys[-2]
            dydt = dy / dt
            self.last_vel = dydt[self.m.n_tris('fault') * 12:]
        if len(ts) % self.cfg.get('refresh_disp_freq', int(1e12)) == 0:
            disp_slip_deficit = self.solve_for_disp_slip_deficit(rk.t, rk.y)
            _, state, _ = self.get_components(rk.y)
            slip_deficit = self.m.get_dofs(disp_slip_deficit, 'fault')
            surf_disp_deficit = self.m.get_dofs(disp_slip_deficit, 'surf')

            plate_dist = rk.t * self.cfg['plate_rate']
            plate_motion = (plate_dist * self.field_inslipdir).reshape(-1)
            slip = self.ones_interior * (plate_motion - slip_deficit)
            surf_disp = -surf_disp_deficit - (self.locked_fault_surf_disp_deficit * plate_dist)
            rk.y = np.concatenate((slip, state, surf_disp))

    def display_model(self, t, y, plotter = plot_fields):
        print(t / siay)
        data = self.solve_for_full_state(t, y)
        disp_slip, state, traction, fault_V, dstatedt, surf_vel = data
        print('slip deficit')
        plotter(self, self.m.get_dofs(disp_slip, 'fault'))
        print('surface displacement deficit')
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

        self.surf_tri_centers = np.mean(self.m.pts[self.m.get_tris('surf')], axis = 1)

        # self.sides = self.which_side(self.surf_tri_centers).astype(np.float)
        # half_slipdir = np.array(self.cfg['slipdir']) / 2.0
        # tri_dispdirs = (
        #     (self.sides[:,np.newaxis] * 2 - 1.0) *
        #     np.tile(half_slipdir[np.newaxis, :], (self.m.n_tris('surf'), 1))
        # )
        # self.surface_dispdirs = np.tile(tri_dispdirs[:,np.newaxis,:], (1,3,1)).flatten()

    @property
    @remember
    def locked_fault_surf_disp_deficit(self):
        return self.m.get_dofs(
            self.slip_to_disp(self.field_inslipdir_interior),
            'surf'
        )


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


def get_slip_to_disp(m, cfg, H):
    base_cs = tct.continuity_constraints(m.pts, m.tris, m.get_start('fault'))
    base_cs.extend(tct.free_edge_constraints(m.tris))

    mass_op = tct.MassOp(
        cfg['tectosaur_cfg']['quad_mass_order'], m.pts, m.tris
    ).mat

    def f(slip):
        cs = base_cs + tct.all_bc_constraints(
            m.n_tris('surf'), m.n_tris(), slip
        )
        cm, c_rhs = tct.build_constraint_matrix(cs, m.n_dofs())

        rhs = -H.dot(c_rhs)
        out = tectosaur_topo.solve.iterative_solve(
            H, cm, rhs, lambda x: x, dict(solver_tol = 1e-4)
        )
        return out + c_rhs
    return f

def get_disp_slip_to_traction(m, cfg, H):
    csS = tct.continuity_constraints(m.pts, m.tris, m.get_start('fault'))
    csF = tct.continuity_constraints(m.pts, m.get_tris('fault'), m.get_end('fault'))
    cs = tct.build_composite_constraints((csS, 0), (csF, m.n_dofs('surf')))
    cs.extend(tct.free_edge_constraints(m.tris))
    cm, c_rhs = tct.build_constraint_matrix(cs, m.n_dofs())

    traction_mass_op = tct.MassOp(cfg['tectosaur_cfg']['quad_mass_order'], m.pts, m.tris)
    constrained_traction_mass_op = cm.T.dot(traction_mass_op.mat.dot(cm))

    def f(disp_slip):
        np.testing.assert_almost_equal(c_rhs, 0.0)
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

def fault_surf_intersection_traction_constraints(m):
    surf_tris = m.get_tris('surf')
    unscaled_ns = tct.util.geometry.unscaled_normals(m.pts[surf_tris])
    ns = tct.util.geometry.normalize(unscaled_ns)
    pt_ns = np.zeros((m.pts.shape[0], 3))
    for i in range(m.n_tris('surf')):
        for d in range(3):
            pt_ns[m.tris[i,d]] += ns[i]
    n_ns = np.ones((m.pts.shape[0]))
    unique, counts = np.unique(surf_tris, return_counts=True)
    n_ns[unique] = counts
    pt_ns /= n_ns[:,np.newaxis]
    cs = []
    for i in m.get_tri_idxs('fault'):
        for d in range(3):
            n = pt_ns[m.tris[i,d]]
            if np.all(n == 0):
                continue
            assert(np.where(surf_tris == m.tris[i,d])[0].shape[0] > 0)
            ts = []
            for d2 in range(3):
                if n[d2] == 0.0:
                    continue
                fault_dof = i * 9 + d * 3 + d2
                ts.append(Term(n[d2], fault_dof))
            cs.append(ConstraintEQ(ts, 0))
    return cs


def get_traction_to_slip(m, cfg, H):
    t = cfg['Timer']()
    csS = tct.continuity_constraints(m.pts, m.tris, m.get_start('fault'))
    csF = tct.continuity_constraints(m.pts, m.get_tris('fault'), m.get_end('fault'))
    cs = tct.build_composite_constraints((csS, 0), (csF, m.n_dofs('surf')))
    cs.extend(tct.free_edge_constraints(m.tris))

    cm, c_rhs = tct.build_constraint_matrix(cs, m.n_dofs())
    cm = cm.tocsr()
    cmT = cm.T.tocsr()
    t.report('t2s -- build constraints')

    traction_mass_op = tct.MassOp(cfg['tectosaur_cfg']['quad_mass_order'], m.pts, m.tris)
    np.testing.assert_almost_equal(c_rhs, 0.0)
    t.report('t2s -- build massop')

    # nearfield_H = H.nearfield.full_scipy_mat_no_correction()
    # diag_H = nearfield_H.diagonal()
    # def prec(x):
    #     return cm.T.dot(cm.dot(x) / diag_H)

    # nearfield_H = H.nearfield.full_scipy_mat_no_correction()
    # constrained_nearfield_H = cmT.dot(nearfield_H.dot(cm))
    # t.report('t2s -- constrained nearfield')
    # spilu = scipy.sparse.linalg.spilu(constrained_nearfield_H)
    # t.report('t2s -- spilu')
    # def prec(x):
    #     return spilu.solve(x)

    # U = build_elastic_op(m, cfg, 'U')
    # nearfield_U = U.nearfield.full_scipy_mat_no_correction()
    # diag_U = nearfield_U.diagonal()
    # def prec(x):
    #     return cmT.dot(U.dot(cm.dot(x)))

    def prec(x):
        return x

    def f(traction):
        rhs = -traction_mass_op.dot(traction / cfg['sm'])
        out = tectosaur_topo.solve.iterative_solve(
            H, cm, rhs, prec, dict(solver_tol = 1e-4)
        )
        return out
    f.H = H
    f.cm = cm
    f.traction_mass_op = traction_mass_op
    return f

def refine_mesh_and_initial_conditions(m, slip_deficit):
    fields = [slip_deficit.reshape((-1,3,3))[:,:,d] for d in range(3)]
    m_refined, new_fields = tct.mesh.refine.refine_to_size(
        (m.pts, m.tris), 0.000000001, recurse = False, fields = fields
    )
    slip_deficit2 = np.swapaxes(np.swapaxes(np.array(new_fields), 0, 2), 0, 1)
    surf_tris = m_refined[1][:m.n_tris('surf') * 4].copy()
    fault_tris = m_refined[1][m.n_tris('surf') * 4:].copy()
    m2 = tct.CombinedMesh.from_named_pieces([
        ('surf', (m_refined[0], surf_tris)),
        ('fault', (m_refined[0], fault_tris))
    ])
    return m2, slip_deficit2
