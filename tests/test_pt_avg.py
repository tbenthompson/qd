import numpy as np

import tectosaur.mesh.mesh_gen as mesh_gen

import cppimport.import_hook
from qd.pt_average import pt_averageD as pt_average_cpp

def pt_average_py(pts, tris, field):
    avg_field = np.empty_like(field)
    for i in range(pts.shape[0]):
        tri_idxs, corner_idxs = np.where(tris == i)
        idxs = tri_idxs * 3 + corner_idxs
        val = np.mean(field[idxs])
        avg_field[idxs] = val
    return avg_field

def test_cpp_pt_average():
    n = 5
    corners = [[-1, -1, 0], [-1, 1, 0], [1, 1, 0], [1, -1, 0]]
    m = mesh_gen.make_rect(n, n, corners)
    x = np.random.rand(m[1].shape[0] * 3)
    y1 = pt_average_py(m[0], m[1], x)
    y2 = pt_average_cpp(m[0], m[1], x)
    np.testing.assert_almost_equal(y1, y2)

