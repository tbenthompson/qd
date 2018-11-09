import tectosaur as tct
from tectosaur.fast_constraints import ConstraintEQ, Term
from qd.topo_model import *

def test_constraint_build():
    import pickle
    with open('tests/cs.pkl', 'rb') as f:
        data = pickle.load(f)
    cs = [
        ConstraintEQ(
            [Term(t[1], t[0]) for t in entry[0]],
            entry[1]
        )
        for entry in data
    ]
    t = tct.Timer()
    cm, c_rhs = tct.build_constraint_matrix(cs, 225000)
    t.report('built')

def test_topo_bug():
    n_fault = 51
    fault_L = 4000
    fault_W = 4000
    fault_m = tct.make_rect(n_fault, n_fault, [
            [-fault_L, 0, 0], [-fault_L, 0, -2 * fault_W],
                [fault_L, 0, -2 * fault_W], [fault_L, 0, 0]
                ])

    n_surf = 101
    surf_L = 8000
    surf_W = 8000
    surf_m = tct.make_rect(n_surf, n_surf, [
            [-surf_L, surf_W, 0], [-surf_L, -surf_W, 0],
                [surf_L, -surf_W, 0], [surf_L, surf_W, 0]
                ])
    n_tris = fault_m[1].shape[0]

    m = tct.CombinedMesh.from_named_pieces([('surf', surf_m), ('fault', fault_m)])

    qd_cfg = dict(
        # Material properties
        sm = 2e10,          # Shear modulus (Pa)
        pr = 0.25,          # Poisson ratio
        density = 2670,     # rock density (kg/m^3)

        # Frictional properties
        Dc = 0.012,         # state evolution length scale (m)
        f0 = 0.6,           # baseline coefficient of friction
        V0 = 1e-6,          # when V = V0, f = f0, V is (m/s)
        a = np.ones(n_tris * 3) * 0.010,
        b = np.ones(n_tris * 3) * 0.015,

        # Boundary conditions
        plate_rate = 1e-9,  # (m/s), equivalent to ~31.5 mm/yr
        slipdir = (1.0, 0.0, 0.0),

        # This is only necessary because this is a full space model and there's no concept of depth or gravity
        additional_normal_stress = 50e6,

        # numerical preferences
        only_x = True, # slip/velocity/traction in the y,z directions are set = 0
        timestep_tol = 1e-3, # error tolerance for the RK45 time stepper

        tectosaur_cfg = dict(
            quad_coincident_order = 8,
            quad_edgeadj_order = 8,
            quad_vertadj_order = 8,
            quad_mass_order = 5,
            quad_far_order = 3,
            quad_near_order = 5,
            quad_near_threshold = 2.5,
            float_type = np.float32,
            use_fmm = False,
            fmm_order = 150,
            fmm_mac = 3.0,
            pts_per_cell = 450,
            log_level = 'DEBUG'
        )
    )

    model = TopoModel(m, qd_cfg)
    # print_length_scales(model)

    cm, c_rhs = surf_fault_continuity(m, False)
    traction_mass_op = tct.MassOp(model.cfg['tectosaur_cfg']['quad_mass_order'], m.pts, m.tris)
    constrained_traction_mass_op = cm.T.dot(traction_mass_op.mat.dot(cm))
    disp_slip = np.zeros(model.H().shape[1])
    import ipdb
    ipdb.set_trace()
    rhs = -model.H().dot(disp_slip)
    constrained_rhs = cm.T.dot(rhs)

    def callback(x):
        callback.iter += 1
        print(callback.iter)
    callback.iter = 0

    soln = cg(constrained_traction_mass_op, constrained_rhs)

