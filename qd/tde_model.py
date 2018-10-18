import numpy as np

from tectosaur.mesh.combined_mesh import CombinedMesh
from .tde_ops import tde_matrix, get_tde_slip_to_traction
from .helpers import tri_normal_info

class TDEModel:
    def __init__(self, m, cfg):
        self.cfg = cfg
        self.setup_mesh(m)
        self.setup_edge_bcs()
        self.calc_derived_constants()

    @property
    def tde_matrix(self):
        if getattr(self, '_tde_matrix', None) is None:
            self._tde_matrix = tde_matrix(self)
        return self._tde_matrix

    @property
    def inv_tde_matrix(self):
        if getattr(self, '_inv_tde_matrix', None) is None:
            self._inv_tde_matrix = np.linalg.inv(self.tde_matrix)
        return self._inv_tde_matrix

    @property
    def slip_to_traction(self):
        if getattr(self, '_slip_to_traction', None) is None:
            self._slip_to_traction = get_tde_slip_to_traction(self.tde_matrix, self.cfg)
        return self._slip_to_traction

    def setup_mesh(self, m):
        if type(m) is CombinedMesh:
            self.m = m
        else:
            self.m = CombinedMesh.from_named_pieces([('fault', m)])
        self.unscaled_normals, self.tri_size, self.tri_normals = \
            tri_normal_info(self.m)
        self.tri_centers = np.mean(self.m.pts[self.m.tris], axis = 1)

        self.n_tris = self.m.tris.shape[0]
        self.basis_dim = 1
        self.n_dofs = self.basis_dim * self.n_tris

    def setup_edge_bcs(self):
        self.ones_interior = np.ones(self.n_tris * 3)
        self.field_inslipdir_interior = np.empty(self.n_tris * 3)
        for d in range(3):
            val = self.cfg.get('slipdir', (1.0, 0.0, 0.0))[d]
            self.field_inslipdir_interior.reshape(-1,3)[:,d] = val

        self.field_inslipdir = self.field_inslipdir_interior.copy()
        self.field_inslipdir_edges = (
            self.field_inslipdir - self.field_inslipdir_interior
        )

    def calc_derived_constants(self):
        # Shear wave speed (m/s)
        self.cfg['cs'] = np.sqrt(self.cfg['sm'] / self.cfg['density'])

        # The radiation damping coefficient (kg / (m^2 * s))
        self.cfg['eta'] = self.cfg['sm'] / (2 * self.cfg['cs'])

    def print_length_scales(self):
        sigma_n = self.cfg['additional_normal_stress']

        self.mesh_L = np.max(np.sqrt(self.tri_size))
        self.Lb = self.cfg['sm'] * self.cfg['Dc'] / (sigma_n * self.cfg['b'])

        #TODO: Remove and replace with empirical version directly from matrix.
        self.hstar = (
            (np.pi * self.cfg['sm'] * self.cfg['Dc']) /
            (sigma_n * (self.cfg['b'] - self.cfg['a']))
        )
        self.hstarRA = (
            (2.0 / np.pi) * self.cfg['sm'] * self.cfg['b'] * self.cfg['Dc']
            / ((self.cfg['b'] - self.cfg['a']) ** 2 * sigma_n)
        )
        self.hstarRA3D = np.pi ** 2 / 4.0 * self.hstarRA

        # all_fields = np.vstack((Lb, hstar, np.ones_like(hstar) * mesh_L)).T
        # plot_fields(m, all_fields)
        print('hstar (2d antiplane, erickson and dunham 2014)', np.min(np.abs(self.hstar)))
        print('hstar_RA (2d antiplane, rubin and ampuero 2005)', np.min(np.abs(self.hstarRA)))
        print('hstar_RA3D (3d strike slip, lapusta and liu 2009)', np.min(np.abs(self.hstarRA3D)))
        print('cohesive zone length scale', np.min(self.Lb))
        print('mesh length scale', self.mesh_L)
