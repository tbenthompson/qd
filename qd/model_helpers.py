import logging
import numpy as np

def print_length_scales(model):
    sigma_n = model.cfg['additional_normal_stress']

    mesh_L = np.max(np.sqrt(model.tri_size))
    Lb = model.cfg['sm'] * model.cfg['Dc'] / (sigma_n * model.cfg['b'])

    #TODO: Remove and replace with empirical version directly from matrix.
    hstar = (
        (np.pi * model.cfg['sm'] * model.cfg['Dc']) /
        (sigma_n * (model.cfg['b'] - model.cfg['a']))
    )
    hstarRA = (
        (2.0 / np.pi) * model.cfg['sm'] * model.cfg['b'] * model.cfg['Dc']
        / ((model.cfg['b'] - model.cfg['a']) ** 2 * sigma_n)
    )
    hstarRA3D = np.pi ** 2 / 4.0 * hstarRA

    # all_fields = np.vstack((Lb, hstar, np.ones_like(hstar) * mesh_L)).T
    # plot_fields(m, all_fields)
    print('hstar (2d antiplane, erickson and dunham 2014)', np.min(np.abs(hstar)))
    print('hstar_RA (2d antiplane, rubin and ampuero 2005)', np.min(np.abs(hstarRA)))
    print('hstar_RA3D (3d strike slip, lapusta and liu 2009)', np.min(np.abs(hstarRA3D)))
    print('cohesive zone length scale', np.min(Lb))
    print('mesh length scale', mesh_L)

def calc_derived_constants(cfg):
    if type(cfg['tectosaur_cfg']['log_level']) is str:
        log_level = getattr(logging, cfg['tectosaur_cfg']['log_level'])
        cfg['tectosaur_cfg']['log_level'] = log_level

    out_cfg = cfg.copy()

    # Shear wave speed (m/s)
    out_cfg['cs'] = np.sqrt(out_cfg['sm'] / out_cfg['density'])

    # The radiation damping coefficient (kg / (m^2 * s))
    out_cfg['eta'] = out_cfg['sm'] / (2 * out_cfg['cs'])
    return out_cfg

def remember(f):
    def g(*args, **kwargs):
        if g.val is None:
            g.val = f(*args, **kwargs)
        return g.val
    g.val = None
    return g
