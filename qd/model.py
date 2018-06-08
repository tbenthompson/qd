import logging
import numpy as np

import tectosaur.mesh.mesh_gen
from tectosaur.mesh.combined_mesh import CombinedMesh
from tectosaur.util.geometry import unscaled_normals
from tectosaur.constraint_builders import free_edge_constraints
from tectosaur.constraints import build_constraint_matrix

from .ops import get_slip_to_traction

class FullspaceModel:
    def __init__(self, m, cfg):
        self.cfg = cfg
        self.setup_mesh(m)
        self.setup_edge_bcs()
        self.calc_derived_constants()
        self.slip_to_traction = get_slip_to_traction(self.m, self.cfg)

    def setup_mesh(self, m):
        self.m = CombinedMesh.from_named_pieces([('fault', m)])

        self.unscaled_tri_normals = unscaled_normals(self.m.pts[self.m.tris])
        self.tri_normals = self.unscaled_tri_normals / np.linalg.norm(self.unscaled_tri_normals, axis = 1)[:, np.newaxis]

    def setup_edge_bcs(self):
        cs = free_edge_constraints(self.m.get_tris('fault'))
        cm, c_rhs = build_constraint_matrix(cs, self.m.n_dofs('fault'))

        constrained_slip = np.ones(cm.shape[1])
        self.ones_interior = cm.dot(constrained_slip)
        self.field_100_interior = self.ones_interior.copy()
        self.field_100_interior.reshape(-1,3)[:,1] = 0.0
        self.field_100_interior.reshape(-1,3)[:,2] = 0.0

        self.field_100 = self.field_100_interior.copy()
        self.field_100.reshape(-1,3)[:,0] = 1.0
        self.field_100_edges = self.field_100 - self.field_100_interior

    def calc_derived_constants(self):
        if type(self.cfg['tectosaur_cfg']['log_level']) is str:
            log_level = getattr(logging, self.cfg['tectosaur_cfg']['log_level'])
            self.cfg['tectosaur_cfg']['log_level'] = log_level

        # Shear wave speed (m/s)
        self.cfg['cs'] = np.sqrt(self.cfg['sm'] / self.cfg['density'])

        # The radiation damping coefficient (kg / (m^2 * s))
        self.cfg['eta'] = self.cfg['sm'] / (2 * self.cfg['cs'])

    def print_length_scales(self):
        sigma_n = self.cfg['additional_normal_stress']

        self.mesh_L = np.max(np.sqrt(np.linalg.norm(
            self.unscaled_tri_normals, axis = 1
        )))
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
