import numpy as np
from qd.derivs import pt_average

def test_pt_average():
    pts = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]])
    tris = np.array([[0,1,2], [2,1,3]])
    field = np.random.rand(6)
    avg_field = pt_average(pts, tris, field)

    correct = [
        field[0],
        (field[1] + field[4]) / 2.0,
        (field[2] + field[3]) / 2.0,
        (field[2] + field[3]) / 2.0,
        (field[1] + field[4]) / 2.0,
        field[5]
    ]
    np.testing.assert_almost_equal(avg_field, correct)
