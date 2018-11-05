import numpy as np

from scipy.sparse.linalg import spsolve

import tectosaur
from tectosaur.mesh.combined_mesh import CombinedMesh
from tectosaur.ops.mass_op import MassOp
from tectosaur.ops.sparse_integral_op import RegularizedSparseIntegralOp
from tectosaur.ops.dense_integral_op import RegularizedDenseIntegralOp
from tectosaur.ops.sparse_farfield_op import FMMFarfieldOp, TriToTriDirectFarfieldOp
from tectosaur.constraint_builders import free_edge_constraints, continuity_constraints
from tectosaur.constraints import build_constraint_matrix
from tectosaur.util.geometry import unscaled_normals
import tectosaur_topo.solve

def setup_slip_traction(m, cfg):
    setup_logging(cfg)
    cm = build_continuity(m, cfg)
    H = build_hypersingular(m, cfg)
    traction_mass_op = MassOp(cfg['tectosaur_cfg']['quad_mass_order'], m.pts, m.tris)
    return H, traction_mass_op, cm

def setup_logging(cfg):
    tectosaur.logger.setLevel(cfg['tectosaur_cfg']['log_level'])
    tectosaur_topo.logger.setLevel(cfg['tectosaur_cfg']['log_level'])

def build_continuity(m, cfg):
    cs = continuity_constraints(m.get_tris('fault'), np.array([]))
    cs.extend(free_edge_constraints(m.get_tris('fault')))
    cm, c_rhs = build_constraint_matrix(cs, m.n_dofs('fault'))
    return cm

def build_hypersingular(m, cfg):
    op_cfg = cfg['tectosaur_cfg']
    return RegularizedSparseIntegralOp(
        op_cfg['quad_coincident_order'],
        op_cfg['quad_edgeadj_order'],
        op_cfg['quad_vertadj_order'],
        op_cfg['quad_far_order'],
        op_cfg['quad_near_order'],
        op_cfg['quad_near_threshold'],
        'elasticRH3', 'elasticRH3', [1.0, cfg['pr']], m.pts, m.tris, op_cfg['float_type'],
        farfield_op_type = get_farfield_op(op_cfg),
        obs_subset = m.get_tri_idxs('fault'),
        src_subset = m.get_tri_idxs('fault')
    )

def get_farfield_op(cfg):
    if cfg['use_fmm']:
        return FMMFarfieldOp(cfg['fmm_mac'], cfg['pts_per_cell'], alpha = cfg['fmm_alpha'])
    else:
        return TriToTriDirectFarfieldOp

def get_slip_to_traction(m, cfg):

    def f(slip):
        t = cfg['Timer']()
        rhs = -f.H.dot(slip)
        t.report('H.dot')
        out = f.cm.dot(spsolve(f.constrained_traction_mass_op, f.cm.T.dot(rhs)))
        t.report('spsolve')

        if cfg.get('only_x', False):
            out.reshape((-1,3))[:,1] = 0.0
            out.reshape((-1,3))[:,2] = 0.0
        out = cfg['sm'] * out
        t.report('return')
        return out

    setup_logging(cfg)
    f.H, f.traction_mass_op, f.cm = setup_slip_traction(m, cfg)
    f.constrained_traction_mass_op = f.cm.T.dot(f.traction_mass_op.mat.dot(f.cm))

    return f

def get_traction_to_slip(m, cfg):
    def f(traction):
        rhs = -f.traction_mass_op.dot(traction / cfg['sm'])
        out = tectosaur_topo.solve.iterative_solve(
            f.H, f.cm, rhs, lambda x: x, dict(solver_tol = 1e-8)
        )
        return out
    f.H, f.traction_mass_op, f.cm = setup_slip_traction(m, cfg)
    return f

