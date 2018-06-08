import numpy as np

from scipy.sparse.linalg import spsolve

import tectosaur
from tectosaur.util.timer import Timer
from tectosaur.mesh.combined_mesh import CombinedMesh
from tectosaur.ops.mass_op import MassOp
from tectosaur.ops.sparse_integral_op import SparseIntegralOp, FMMFarfieldBuilder
from tectosaur.constraint_builders import free_edge_constraints, continuity_constraints
from tectosaur.constraints import build_constraint_matrix
from tectosaur.util.geometry import unscaled_normals
import tectosaur_topo.solve


def make_integral_op(m, k_name, k_params, cfg, name1, name2):
    if cfg['use_fmm']:
        farfield = FMMFarfieldBuilder(
            cfg['fmm_order'], cfg['fmm_mac'], cfg['pts_per_cell']
        )
    else:
        farfield = None
    return SparseIntegralOp(
        cfg['quad_vertadj_order'], cfg['quad_far_order'],
        cfg['quad_near_order'], cfg['quad_near_threshold'],
        k_name, k_params, m.pts, m.tris, cfg['float_type'],
        farfield_op_type = farfield,
        obs_subset = m.get_tri_idxs(name1),
        src_subset = m.get_tri_idxs(name2)
    )

def make_mass_op(m, cfg):
    return MassOp(cfg['quad_mass_order'], m.pts, m.tris)

def get_slip_to_traction(m, qd_cfg):
    tectosaur.logger.setLevel(qd_cfg['tectosaur_cfg']['log_level'])
    tectosaur_topo.logger.setLevel(qd_cfg['tectosaur_cfg']['log_level'])
    cs2 = continuity_constraints(m.get_tris('fault'), np.array([]))
    cs2.extend(free_edge_constraints(m.get_tris('fault')))
    cm2, c_rhs2 = build_constraint_matrix(cs2, m.n_dofs('fault'))
    hypersingular_op = make_integral_op(m, 'elasticH3', [1.0, qd_cfg['pr']], qd_cfg['tectosaur_cfg'], 'fault', 'fault')
    traction_mass_op = make_mass_op(m, qd_cfg['tectosaur_cfg'])
    constrained_traction_mass_op = cm2.T.dot(traction_mass_op.mat.dot(cm2))
    def slip_to_traction(slip):
        t = Timer()
        rhs = -hypersingular_op.dot(slip)
        t.report('H.dot')
        out = cm2.dot(spsolve(constrained_traction_mass_op, cm2.T.dot(rhs)))

        if qd_cfg['only_x']:
            out.reshape((-1,3))[:,1] = 0.0
            out.reshape((-1,3))[:,2] = 0.0
        return qd_cfg['sm'] * out
    return slip_to_traction

def get_traction_to_slip(m, qd_cfg):
    tectosaur.logger.setLevel(qd_cfg['tectosaur_cfg']['log_level'])
    tectosaur_topo.logger.setLevel(qd_cfg['tectosaur_cfg']['log_level'])
    cs2 = continuity_constraints(m.get_tris('fault'), np.array([]))
    cs2.extend(free_edge_constraints(m.get_tris('fault')))
    cm2, c_rhs2 = build_constraint_matrix(cs2, m.n_dofs('fault'))
    hypersingular_op = make_integral_op(m, 'elasticH3', [1.0, qd_cfg['pr']], qd_cfg['tectosaur_cfg'], 'fault', 'fault')
    traction_mass_op = make_mass_op(m, qd_cfg['tectosaur_cfg'])
    #constrained_traction_mass_op = cm2.T.dot(traction_mass_op.mat.dot(cm2))
    def traction_to_slip(traction):
        rhs = -traction_mass_op.dot(traction / qd_cfg['sm'])
        out = tectosaur_topo.solve.iterative_solve(
            hypersingular_op, cm2, rhs, lambda x: x, dict(solver_tol = 1e-8)
        )
        return out
    return traction_to_slip
