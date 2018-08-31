import numpy as np

def dofs_to_pts(model, field):
    if model.basis_dim == 1:
        return constant_basis_to_pts(model.m.pts, model.m.tris, field)
    else:
        return linear_basis_to_pts(model.m.pts, model.m.tris, field)

def linear_basis_to_pts(pts, tris, field):
    n_fields = field.shape[1]
    pt_field = np.empty((pts.shape[0], n_fields))
    pt_field[tris] = field.reshape((-1, 3, n_fields))
    return pt_field

def constant_basis_to_pts(pts, tris, field):
    n_fields = field.shape[1]
    pt_idxs, counts = np.unique(tris, return_counts = True)
    pt_counts = np.ones(pts.shape[0], dtype = np.int32)
    pt_counts[pt_idxs] = counts
    pt_field = np.zeros((pts.shape[0], n_fields))
    for i in range(tris.shape[0]):
        pt_field[tris[i]] += field[i]
    pt_field /= pt_counts[:, np.newaxis]
    return pt_field

def constant_to_linear(model, field):
    pt_vals = dofs_to_pts(model, field)
    dofs = np.empty((model.m.tris.shape[0], 3, field.shape[-1]))
    for i in range(model.m.tris.shape[0]):
        for d in range(3):
            dofs[i, d] = pt_vals[model.m.tris[i, d]]
    return dofs
