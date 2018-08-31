import numpy as np

from tectosaur.util.geometry import unscaled_normals

def tri_normal_info(m):
    unscaled_tri_normals = unscaled_normals(m.pts[m.tris])
    tri_size = np.linalg.norm(unscaled_tri_normals, axis = 1)
    tri_normals = unscaled_tri_normals / tri_size[:, np.newaxis]
    return unscaled_tri_normals, tri_size, tri_normals
