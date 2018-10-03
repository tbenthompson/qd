import numpy as np
import matplotlib.pyplot as plt
import tectosaur as tct

from qd.ops_nonplanar import build_terms, build_system, solve_system

def gaussian(a, b, c, x):
    # return np.ones_like(x)
    return a * np.exp(-((x - b) ** 2) / (2 * c ** 2))

def get_slip_field(pts, tris):
    gauss_params = (1.0, 0.0, 0.3)
    gauss_center = (0.0, 0.0, 0.0)
    n_tris = tris.shape[0]
    tri_pts = pts[tris].reshape((-1,3))
    dist = np.linalg.norm(tri_pts - gauss_center, axis = 1)
    strike_slip = gaussian(*gauss_params, dist)
    slip = np.zeros((strike_slip.shape[0], 3))
    slip[:,0] = strike_slip
    return slip.flatten()

def plot_tectosaur(m, field, name, show = True):
    pts, tris = m
    pt_field = np.empty(pts.shape[0])
    pt_field[tris] = field

    plt.figure()
    cmap = 'PuOr'
    plt.tricontourf(
        pts[:,0], pts[:,2], tris,
        pt_field, cmap = cmap,
        # levels = f_levels,
        extend = 'both'
    )
    plt.colorbar()
    plt.show()

tectosaur_cfg = dict(
    quad_coincident_order = 8,
    quad_edgeadj_order = 8,
    quad_mass_order = 3,
    quad_vertadj_order = 6,
    quad_far_order = 3,
    quad_near_order = 5,
    quad_near_threshold = 2.5,
    float_type = np.float64,
    use_fmm = False,
    fmm_order = 150,
    fmm_mac = 3.0,
    pts_per_cell = 450,
    sm = 1.0,
    pr = 0.25
)

def make_mesh(n):
    corners = [[-1, 0, 1], [-1, 0, -1], [1, 0, -1], [1, 0, 1]]
    return tct.make_rect(n, n, corners)
#
# def test_convergence_co():
#     pts, tris = make_mesh(10)
#     from tectosaur.nearfield.nearfield_op import PairsIntegrator
#     Ks = [
#         'elasticU3',
#         'elasticRT3',
#         'elasticRA3',
#         'elasticRH3'
#     ]
#     for K in Ks:
#         for i in range(5):
#             print(K)
#         pairs_int = PairsIntegrator(
#             K, [1.0, 0.25], np.float32, 2, 5, pts, tris
#         )
#         co_tris = np.arange(tris.shape[0])
#         co_indices = np.array([co_tris, co_tris]).T.copy()
#
#         for nq in range(1, 11):
#
#             co_mat1 = pairs_int.coincident(nq, co_indices)
#             co_mat2 = pairs_int.coincident(nq + 1, co_indices)
#             diff = co_mat1 - co_mat2
#             err = np.abs(diff / co_mat2)
#             err[np.isnan(err)] = 1e-15
#
#             l2_diff =  np.sqrt(np.sum(diff ** 2))
#             l2_mat = np.sqrt(np.sum(co_mat2 ** 2))
#             print(nq, co_mat1.flatten()[0], co_mat2.flatten()[0], l2_diff / l2_mat)



def test_compare_terms():
    pts, tris = make_mesh(10)

    tectosaur_cfg['float_type'] = np.float64

    cfg1 = tectosaur_cfg.copy()

    cfg2 = tectosaur_cfg.copy()
    for k,v in cfg2.items():
        if 'order' in k:
            print(k)
            cfg2[k] = v + 1

    terms1 = build_terms(pts, tris, cfg1)
    terms2 = build_terms(pts, tris, cfg2)

    Unear1 = terms1['U'].ops[0].nearfield.full_scipy_mat_no_correction().todense()
    Unear2 = terms2['U'].ops[0].nearfield.full_scipy_mat_no_correction().todense()

    plt.figure()
    plt.imshow(Unear1)
    plt.colorbar()

    plt.figure()
    plt.imshow(Unear2)
    plt.colorbar()

    diff = Unear1 - Unear2
    percent_err = np.abs(diff) / np.abs(Unear2)
    log_percent = np.log10(percent_err)
    log_percent[np.isnan(log_percent)] = -10

    plt.figure()
    plt.imshow(np.log10(np.abs(diff)))
    plt.colorbar()

    # plt.figure()
    # plt.imshow(percent_err)
    # plt.colorbar()

    # plt.figure()
    # plt.imshow(log_percent)
    # plt.colorbar()

    plt.show()

def test_dbem():
    pts, tris = make_mesh(20)

    slip = get_slip_field(pts, tris)
    # plot_tectosaur(
    #     (pts, tris), slip.reshape(-1,3)[:,0].reshape(-1,3), 'inputslip'
    # )
    terms = build_terms(pts, tris, tectosaur_cfg)
    system = build_system(pts, tris, terms, slip)
    traction = solve_system(system, 1e-4)

    sxy = traction.reshape(-1,3,3)[:,:,0]
    syz = traction.reshape(-1,3,3)[:,:,2]
    plot_tectosaur((pts, tris), sxy, 'sxy')
    plot_tectosaur((pts, tris), syz, 'syz')
