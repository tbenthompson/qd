import numpy as np
import scipy.sparse.linalg
import cutde.fullspace
from .helpers import tri_normal_info

def build_xyz_slip_inputs(model):
    tensile_slip_vec = model.tri_normals
    dip_slip_vec = np.cross([1,0,0], tensile_slip_vec)
    strike_slip_vec = np.cross(tensile_slip_vec, dip_slip_vec)
    slip = np.zeros((model.n_tris, 3, 3))
    for d in range(3):
        v = np.zeros(3)
        v[d] = 1.0
        slip[:, d, 0] = strike_slip_vec.dot(v)
        slip[:, d, 1] = dip_slip_vec.dot(v)
        slip[:, d, 2] = tensile_slip_vec.dot(v)
    slip = slip.reshape((-1, 3))
    return slip

def tde_stress_matrix(model):
    tri_pts = model.m.pts[model.m.tris]
    tri_pts_3 = np.repeat(tri_pts, 3, axis = 0)
    slip = build_xyz_slip_inputs(model)
    all_strains = cutde.fullspace.clu_strain_all_pairs(
        model.tri_centers, tri_pts_3, slip, model.cfg['pr']
    )
    stress = cutde.fullspace.strain_to_stress(
        all_strains.reshape((-1, 6)),
        model.cfg['sm'], model.cfg['pr']
    ).reshape((model.n_tris, 3 * model.n_tris, 6))
    return stress

def stress_to_traction(normals, stress):
    # map from 6 component symmetric to 9 component full tensor
    components = [
        [0, 3, 4],
        [3, 1, 5],
        [4, 5, 2]
    ]
    traction = np.array([
        np.sum([
            stress[:,:,components[i][j]] * normals[:,j,np.newaxis]
            for j in range(3)
        ], axis = 0) for i in range(3)
    ])
    traction = np.swapaxes(traction, 0, 1)
    traction = traction.reshape((model.n_tris * 3, model.n_tris * 3))
    return traction

def tde_matrix(model):
    stress = tde_stress_matrix(model)
    return stress_to_traction(model.tri_normals, stress)

def get_tde_slip_to_traction(tde_matrix, qd_cfg):
    def slip_to_traction(slip):
        out = tde_matrix.dot(slip)
        if qd_cfg['only_x']:
            out.reshape((-1,3))[:,1] = 0.0
            out.reshape((-1,3))[:,2] = 0.0
        return out
    return slip_to_traction

def get_tde_traction_to_slip_iterative(tde_matrix):
    solver_tol = 1e-7
    def traction_to_slip(traction):
        return scipy.sparse.linalg.gmres(
            tde_matrix, traction, tol = solver_tol,
            restart = 500
        )[0]
    return traction_to_slip

def get_tde_traction_to_slip_direct(tde_matrix):
    inverse_tde_matrix = np.linalg.inv(tde_matrix)
    def traction_to_slip(traction):
        return inverse_tde_matrix.dot(traction)
    return traction_to_slip

def get_tde_traction_to_slip(tde_matrix):
    return get_tde_traction_to_slip_iterative(tde_matrix)
