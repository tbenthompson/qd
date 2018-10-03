import numpy as np
import tectosaur as tct
import tectosaur_topo
import scipy.sparse

def build_terms(pts, tris, tectosaur_cfg):
    all_tri_idxs = np.arange(tris.shape[0])
    ops = dict()
    cfg = tectosaur_cfg
    ops['M'] = tct.MassOp(tectosaur_cfg['quad_mass_order'], pts, tris)

    for K in [
            #'U',
            #'T',
            #'A', 'H'
            ]:
        ops[K] = tct.SparseIntegralOp(
            cfg['quad_vertadj_order'], cfg['quad_far_order'],
            cfg['quad_near_order'], cfg['quad_near_threshold'],
            'elastic' + K + '3', [tectosaur_cfg['sm'], tectosaur_cfg['pr']],
            pts, tris, cfg['float_type'],
            farfield_op_type = tct.PtToPtDirectFarfieldOp,
            obs_subset = all_tri_idxs,
            src_subset = all_tri_idxs
        )

    for Kabbr, K, mass_factor in [
            ('U', 'elasticU3', 0.0),
            ('T', 'elasticRT3', -0.5),
            ('A', 'elasticRA3', 0.5),
            ('H', 'elasticRH3', 0.0)
            ]:
        main_op = tct.RegularizedSparseIntegralOp(
            cfg['quad_coincident_order'],
            cfg['quad_edgeadj_order'],
            cfg['quad_vertadj_order'],
            cfg['quad_far_order'],
            cfg['quad_near_order'],
            cfg['quad_near_threshold'],
            K, K, [tectosaur_cfg['sm'], tectosaur_cfg['pr']],
            pts, tris, cfg['float_type'],
            farfield_op_type = tct.TriToTriDirectFarfieldOp,
            obs_subset = all_tri_idxs,
            src_subset = all_tri_idxs
        )
        ops[Kabbr] = tct.SumOp([main_op, tct.MultOp(ops['M'], mass_factor)])
    return ops

# Galerkin dual boundary element method
def gdbem(ops):
    S = tct.SumOp
    N = tct.NegOp
    Mul = tct.MultOp
    U = ops['U']
    T = ops['T']
    A = ops['A']
    H = ops['H']
    M = ops['M']

    # VERSION 2: skip two rows since they are redundant given the constraints
    # on side A: apply displacement integral equation
    # on side B: apply traction integral equation

    # T and A flip sign with normal vector flip
    # on side A -- eq1 = K_U*t_A + K_U*t_B - (K_T+M)*u_A + K_T*u_B
    # on side B -- eq2 = K_A*t_A + (K_A+M)*t_B - K_H*u_A + K_H*u_B
    #
    # v = [
    #    t_A, t_B, u_A, u_B
    # ]
    mat = [
        [N(U), N(U), S([T,M]), N(T)],
        [N(U), N(U), S([T,M]), N(T)],
        [A, S([A,N(M)]), H, N(H)],
        [N(A), S([N(A),M]), N(H), H],
    ]

    P = jacobi_prec
    prec_mat = [
        [N(P(U)),0,0,0],
        [0,N(P(U)),0,0],
        [0,0,N(P(H)),0],
        [0,0,0,P(H)],
    ]
    return mat, prec_mat

def spilu_prec(iop):
    print('spilu!')
    M_scipy = sum([iop.ops[0].nearfield.mat_no_correction.to_bsr().to_scipy()])
    P = scipy.sparse.linalg.spilu(M_scipy)
    class Wrapper:
        def __init__(self, m):
            self.m = m

        def dot(self, x):
            return self.m.solve(x)
    return Wrapper(P)

def jacobi_prec(iop):
    diag = np.sum([
        m.to_bsr().to_scipy().diagonal()
        for m in iop.ops[0].nearfield.mat_no_correction
    ], axis = 0)
    return scipy.sparse.spdiags([1.0 / diag], [0], iop.shape[0], iop.shape[1])

def build_system(pts, tris, ops, slip):
    trac_cs = []
    trac_cs.extend(tct.jump_constraints(np.zeros_like(slip), True))

    disp_cs = []
    disp_cs.extend(tct.free_edge_constraints(tris))
    disp_cs.extend(tct.continuity_constraints(tris, np.array([])))
    disp_cs.extend(tct.jump_constraints(slip, False))

    ND = tris.shape[0] * 9
    n_total_dofs = ND * 4
    cs = tct.build_composite_constraints(
        (trac_cs, 0), (disp_cs, 2 * ND)
    )
    cm, c_rhs = tct.build_constraint_matrix(cs, n_total_dofs)

    chunk_mat, prec_chunk_mat = gdbem(ops)
    lhs = chunks_to_composite_op(chunk_mat, ND)
    rhs = -lhs.dot(c_rhs)

    prec_lhs = chunks_to_composite_op(prec_chunk_mat, ND)
    prec = lambda x: cm.T.dot(prec_lhs.dot(cm.dot(x)))

    return cm, lhs, rhs, prec

def chunks_to_composite_op(chunk_mat, ND):
    ops_and_starts = []
    for i in range(4):
        for j in range(4):
            chunk = chunk_mat[i][j]
            if type(chunk) is int and chunk == 0:
                continue
            ops_and_starts.append((chunk, i * ND, j * ND))
    return tct.CompositeOp(*ops_and_starts)

def solve_system(system, tol):
    cm, lhs, rhs, prec = system
    cfg = dict(solver_tol = tol)
    out = tectosaur_topo.iterative_solve(lhs, cm, rhs, prec, cfg)
    return out.reshape(4, -1, 3)[0, :, :]

# def get_slip_to_traction(m, qd_cfg):
#     def slip_to_traction(slip):
#     return slip_to_traction
