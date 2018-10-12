import numpy as np
import tectosaur as tct
import tectosaur_topo
from scipy.sparse.linalg import spsolve

def get_slip_to_stress(m, qd_cfg):
    tct.logger.setLevel(qd_cfg['tectosaur_cfg']['log_level'])
    cs2 = tct.continuity_constraints(m.get_tris('fault'), np.array([]))
    cs2.extend(tct.free_edge_constraints(m.get_tris('fault')))
    cm2, c_rhs2 = tct.build_constraint_matrix(cs2, m.n_dofs('fault'))

    cfg = qd_cfg['tectosaur_cfg']
    H = [
        tct.RegularizedSparseIntegralOp(
            cfg['quad_coincident_order'],
            cfg['quad_edgeadj_order'],
            cfg['quad_vertadj_order'],
            cfg['quad_far_order'],
            cfg['quad_near_order'],
            cfg['quad_near_threshold'],
            'elasticRH3', 'elasticRH3', [cfg['sm'], cfg['pr']],
            m.pts, m.tris, cfg['float_type'],
            farfield_op_type = tct.TriToTriDirectFarfieldOp,
            obs_subset = m.get_tri_idxs('fault'),
            src_subset = m.get_tri_idxs('fault'),
            force_normal = n
        ) for n in [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0)
        ]
    ]

    stress_mass_op = tct.MassOp(qd_cfg['tectosaur_cfg']['quad_mass_order'], m.pts, m.tris)
    constrained_stress_mass_op = cm2.T.dot(stress_mass_op.mat.dot(cm2))

    def slip_to_stress(slip):
        return [
            cfg['sm'] * cm2.dot(spsolve(
                constrained_stress_mass_op,
                cm2.T.dot(H[d].dot(slip))
            ))
            for d in range(3)
        ]
    return slip_to_stress, (cm2, constrained_stress_mass_op, H)
