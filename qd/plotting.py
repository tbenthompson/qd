import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import clear_output
from mpl_toolkits.axes_grid1 import make_axes_locatable

from . import siay
from .data import skip_existing_prefixed_folders
from .basis_convert import dofs_to_pts

def plot_fields(model, field, which = 'fault', levels = None, cmap = 'seismic',
        symmetric_scale = False, ds = None, figsize = None, dims = [0,2],
        xlim = None, ylim = None):

    field_reshape = field.reshape((model.m.n_tris(which) * 3, -1))
    n_fields = field_reshape.shape[1]

    if figsize is None:
        figsize = (6 * n_fields,5)
    plt.figure(figsize = figsize)

    which_tris = model.m.get_tris(which)
    plot_f = dofs_to_pts(model.m.pts, which_tris, model.basis_dim, field_reshape)
    which_pts_idxs = np.unique(which_tris)
    which_pts = model.m.pts[which_pts_idxs]
    for d in (range(n_fields) if ds is None else ds):
        plt.subplot(1, n_fields, d + 1)

        f_levels = levels
        if f_levels is None:
            f_levels = get_levels(plot_f[which_pts_idxs,d], symmetric_scale)

        cntf = plt.tricontourf(
            model.m.pts[:,dims[0]], model.m.pts[:,dims[1]], which_tris, plot_f[:,d],
            cmap = cmap, levels = f_levels, extend = 'both'
        )
        if xlim is None:
            plt.xlim([np.min(which_pts[:,dims[0]]), np.max(which_pts[:,dims[0]])])
        else:
            plt.xlim(xlim)
        if ylim is None:
            plt.ylim([np.min(which_pts[:,dims[1]]), np.max(which_pts[:,dims[1]])])
        else:
            plt.ylim(ylim)
        plt.colorbar(cntf)
    plt.tight_layout()
    plt.show()

def get_levels(f, symmetric_scale):
    min_f = np.min(f)
    max_f = np.max(f)
    scale_f = np.max(np.abs(f))
    if scale_f == 0.0:
        scale_f = 1.0
    min_f -= 1e-13 * scale_f
    max_f += 1e-13 * scale_f
    if symmetric_scale:
        min_f = -max_f
    return np.linspace(min_f, max_f, 21)

class QDPlotData:
    def __init__(self, data):
        self.data = data
        self.model = self.data.model
        self.cfg = self.data.cfg
        self.t = self.data.ts
        self.y = self.data.ys
        self.t_years = self.t / siay

        self.n_steps = self.y.shape[0]
        self.min_state = [0] * self.n_steps
        self.max_V = [0] * self.n_steps
        self.slip = [0] * self.n_steps
        self.state = [0] * self.n_steps
        self.V = [0] * self.n_steps
        self.dt = [0] * self.n_steps
        for i in range(0, self.n_steps):
            components = self.model.get_components(self.y[i])
            self.slip[i] = components[0]
            self.state[i] = components[1]
            self.min_state[i] = np.min(self.state[i])
            if i > 0:
                self.dt[i] = self.t[i] - self.t[i - 1]
                self.V[i] = (self.slip[i] - self.slip[i - 1]) / self.dt[i]
            self.max_V[i] = np.max(self.V[i])

    def simple_data_file(self, filename):
        slip_at_pts = []
        state_at_pts = []
        for i in range(len(qdp.slip)):
            slip_at_pts.append(qd.dofs_to_pts(
                self.model.m.pts, self.model.m.tris, self.model.basis_dim,
                self.slip[i].reshape((-1,3))
            ))
            state_at_pts.append(qd.dofs_to_pts(
                self.model.m.pts, self.model.m.tris, self.model.basis_dim,
                self.state[i].reshape((-1,1))
            ))
        slip_at_pts = np.array(slip_at_pts)
        state_at_pts = np.array(state_at_pts)[:,:,0]
        np.save(
            filename,
            (
                self.model.m.pts, self.model.m.tris,
                self.t, slip_at_pts, state_at_pts
            )
        )

    def summary(self):
        print('plotting', np.max(self.t_years), 'years and', self.n_steps, 'time steps of data')
        plt.figure(figsize = (16,16))
        plt.subplot(221)
        plt.plot(self.t_years[1:], self.min_state[1:])
        plt.xlabel('t')
        plt.ylabel('$\min(\Psi)$')
        plt.subplot(222)
        plt.plot(self.t_years[1:], np.log10(np.abs(self.max_V[1:]) + 1e-40))
        plt.xlabel('t')
        plt.ylabel('$\log_{10}(\max(\|V_x\|))$')
        plt.subplot(223)
        plt.plot(self.t_years[1:])
        plt.xlabel('step')
        plt.ylabel('t')
        plt.subplot(224)
        plt.plot(np.log10(self.dt[1:]))
        plt.xlabel('step')
        plt.ylabel('$\log_{10}(\Delta t)$')
        plt.show()

    def nicefig(self, field, levels, contour_levels, cmap,
            t_years = None, filepath = None, figsize = (10,8),
            which = 'fault', dim = [0,2]):
        contour_levels = levels[::3]

        is_tde = field.size == self.model.m.n_tris(which)

        fig = plt.figure(figsize = figsize)
        ax = plt.gca()
        div = make_axes_locatable(ax)
        cax = div.append_axes('right', '5%', '5%')

        which_tris = self.model.m.get_tris(which).copy()
        pt_field = dofs_to_pts(
            self.model.m.pts, which_tris,
            self.model.basis_dim, field.reshape(-1, 1)
        )[:,0]
        which_pts_idxs = np.unique(which_tris)
        which_pts = self.model.m.pts[which_pts_idxs]

        color_plot = ax.tricontourf(
            self.model.m.pts[:,dim[0]], self.model.m.pts[:,dim[1]], which_tris,
            pt_field, cmap = cmap, levels = levels, extend = 'both'
        )
        ax.tricontour(
            self.model.m.pts[:,dim[0]], self.model.m.pts[:,dim[1]], which_tris,
            pt_field, levels = contour_levels, extend = 'both',
            linestyles = 'solid', linewidths = 0.75,
            colors = ['#FFFFFF'] * contour_levels.shape[0]
        )

        minpt = np.min(which_pts, axis = 0)
        maxpt = np.max(which_pts, axis = 0)
        width = maxpt - minpt

        F = 0.03
        ax.set_xlim([
            minpt[dim[0]] - width[dim[0]] * F,
            maxpt[dim[0]] + width[dim[0]] * F
        ])
        ax.set_ylim([
            minpt[dim[1]] - width[dim[1]] * F,
            maxpt[dim[1]] + width[dim[1]] * F
        ])
        ax.set_aspect('equal', adjustable='box')

        text_pos = (
            minpt[dim[0]],
            maxpt[dim[1]] + (maxpt[dim[1]] - minpt[dim[1]]) * 0.003
        )
        if t_years is not None:
            ax.text(text_pos[0], text_pos[1], '%.9f' % t_years)

        fig.colorbar(color_plot, cax = cax)
        if filepath is None:
            plt.show()
        else:
            plt.savefig(filepath, bbox_inches = 'tight', dpi = 150)
        plt.close()


    def qd_video(self, steps_to_plot, field_data_fnc, video_prefix = None,
            **kwargs):

        def get_frame_name(frame_idx, n_frames):
            digits = len(str(n_frames))
            return '%0*d' % (digits, frame_idx)

        if video_prefix is not None:
            video_name = skip_existing_prefixed_folders(video_prefix)
            os.makedirs(video_name)

        n_frames = len(steps_to_plot)
        for frame_idx in range(n_frames):
            step = steps_to_plot[frame_idx]
            print('step =', step)
            print('t (yrs) =', self.t_years[step])
            print('dt (secs) =', self.dt[step])
            print('max(V_x) =', self.max_V[step])
            print('min(state) =', self.min_state[step])
            clear_output(wait = True)

            field_data = field_data_fnc(step)
            frame_name = get_frame_name(frame_idx, n_frames)
            filepath = None if video_prefix is None else f'{video_name}/{frame_name}.png'
            self.nicefig(
                *field_data, filepath = filepath,
                t_years = self.t_years[step],
                **kwargs
            )

        return video_name

    def V_info(self, step):
        field = np.log10(np.abs(self.V[step].reshape(-1,3)[:,0] + 1e-40))
        levels = np.linspace(-10,-1,11)
        contour_levels = levels[::3]
        cmap = 'viridis'
        return field, levels, contour_levels, cmap

    def state_info(self, step):
        field = self.state[step]
        levels = np.array(np.linspace(0.6, 0.75, 11).tolist() + [0.8, 1.0])
        contour_levels = levels[::3]
        cmap = 'viridis_r'
        return field, levels, contour_levels, cmap

    def slip_info(step):
        field = slip[step].reshape(-1,3)[:,0]
        levels = np.linspace(0.0, 1.0, 16)
        contour_levels = levels
        cmap = 'viridis'
        return field, levels, contour_levels, cmap

def make_mp4(video_name, framerate = 10):
    # get the number of digits in the image filenames
    digits = len(os.path.splitext(os.listdir(video_name)[0])[0])

    cmd = [
        'ffmpeg', '-framerate', str(framerate),
        '-i', f'{video_name}/%0{digits}d.png',
        '-c:v', 'libx264',
        '-r', '30',
        '-y', '-v', '32',
        video_name + '.mp4'
    ]
    print('running', '"' + ' '.join(cmd) + '"')
    for line in execute(cmd):
        print(line, end = '')

def execute(cmd):
    popen = subprocess.Popen(cmd, stderr = subprocess.PIPE, universal_newlines = True)
    for stdout_line in iter(popen.stderr.readline, ""):
        yield stdout_line
    popen.stderr.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)
