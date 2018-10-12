import numpy as np

from scipy.sparse.linalg import spsolve

import tectosaur
from tectosaur.util.timer import Timer
from tectosaur.mesh.combined_mesh import CombinedMesh
from tectosaur.ops.mass_op import MassOp
from tectosaur.ops.sparse_integral_op import SparseIntegralOp, \
    RegularizedSparseIntegralOp
from tectosaur.ops.dense_integral_op import RegularizedDenseIntegralOp
from tectosaur.ops.sparse_farfield_op import PtToPtFMMFarfieldOp, \
    PtToPtDirectFarfieldOp, TriToTriDirectFarfieldOp
from tectosaur.constraint_builders import free_edge_constraints, continuity_constraints
from tectosaur.constraints import build_constraint_matrix
from tectosaur.util.geometry import unscaled_normals
import tectosaur_topo.solve

def get_slip_to_traction(m, qd_cfg):
    tectosaur.logger.setLevel(qd_cfg['tectosaur_cfg']['log_level'])
    tectosaur_topo.logger.setLevel(qd_cfg['tectosaur_cfg']['log_level'])
    cs2 = continuity_constraints(m.get_tris('fault'), np.array([]))
    cs2.extend(free_edge_constraints(m.get_tris('fault')))
    cm2, c_rhs2 = build_constraint_matrix(cs2, m.n_dofs('fault'))

    op_cfg = qd_cfg['tectosaur_cfg']
    hypersingular_op = RegularizedSparseIntegralOp(
        op_cfg['quad_coincident_order'],
        op_cfg['quad_edgeadj_order'],
        op_cfg['quad_vertadj_order'],
        op_cfg['quad_far_order'],
        op_cfg['quad_near_order'],
        op_cfg['quad_near_threshold'],
        'elasticRH3', 'elasticRH3', [1.0, qd_cfg['pr']], m.pts, m.tris, op_cfg['float_type'],
        farfield_op_type = TriToTriDirectFarfieldOp,
        obs_subset = m.get_tri_idxs('fault'),
        src_subset = m.get_tri_idxs('fault')
    )
    traction_mass_op = MassOp(op_cfg['quad_mass_order'], m.pts, m.tris)
    constrained_traction_mass_op = cm2.T.dot(traction_mass_op.mat.dot(cm2))

    def slip_to_traction(slip):
        # t = Timer()
        rhs = -hypersingular_op.dot(slip)
        # t.report('H.dot')
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

    op_cfg = qd_cfg['tectosaur_cfg']
    hypersingular_op = RegularizedSparseIntegralOp(
        op_cfg['quad_coincident_order'],
        op_cfg['quad_edgeadj_order'],
        op_cfg['quad_vertadj_order'],
        op_cfg['quad_far_order'],
        op_cfg['quad_near_order'],
        op_cfg['quad_near_threshold'],
        'elasticRH3', 'elasticRH3', [1.0, qd_cfg['pr']], m.pts, m.tris, op_cfg['float_type'],
        farfield_op_type = TriToTriDirectFarfieldOp,
        obs_subset = m.get_tri_idxs('fault'),
        src_subset = m.get_tri_idxs('fault')
    )
    traction_mass_op = MassOp(op_cfg['quad_mass_order'], m.pts, m.tris)
    def traction_to_slip(traction):
        rhs = -traction_mass_op.dot(traction / qd_cfg['sm'])
        out = tectosaur_topo.solve.iterative_solve(
            hypersingular_op, cm2, rhs, lambda x: x, dict(solver_tol = 1e-8)
        )
        return out
    return traction_to_slip

