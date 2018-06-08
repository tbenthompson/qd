import os
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import clear_output
from mpl_toolkits.axes_grid1 import make_axes_locatable

from qd_helpers import QDMeshData, separate_slip_state

def plot_fields(m, field, levels = None, cmap = 'seismic', symmetric_scale = False, ds = None, figsize = None):
    field_reshape = field.reshape(m.tris.shape[0],3,-1)
    n_fields = field_reshape.shape[2]
    if figsize is None:
        figsize = (6 * n_fields,5)
    plt.figure(figsize = figsize)
    if ds is None:
        ds = range(n_fields)
    for d in ds:
        plt.subplot(1, n_fields, d + 1)
        pt_field = np.empty((m.pts.shape[0], n_fields))
        pt_field[m.tris] = field_reshape
        f = pt_field[:,d]
        f_levels = levels
        if f_levels is None:
            min_f = np.min(f)
            max_f = np.max(f)
            scale_f = np.max(np.abs(f))
            if scale_f == 0.0:
                scale_f = 1.0
            min_f -= 1e-13 * scale_f
            max_f += 1e-13 * scale_f
            if symmetric_scale:
                min_f = -max_f
            f_levels = np.linspace(min_f, max_f, 21)
        cntf = plt.tricontourf(m.pts[:,0], m.pts[:,2], m.tris, f, cmap = cmap, levels = f_levels, extend = 'both')
        #cntf = plt.tricontour(m.pts[:,0], m.pts[:,2], m.tris, f, levels = f_levels)
        #plt.tricontour(m.pts[:,0], m.pts[:,2], m.tris, f, levels = levels, extend = 'both', linestyles = 'solid', linewidths = 0.75, colors = '#333333')
        #plt.axis('equal')
        plt.colorbar(cntf)
        #plt.axis('equal')
    plt.show()

def plot_setting(t, y, qdm, qd_cfg, slip_to_traction):
    slip, slip_deficit, state, traction, V, dstatedt = solve_for_full_state(
        qdm, qd_cfg, slip_to_traction, t, y
    )
    #print('slip')
    #plot_signs(slip)
    #plot_fields(np.log10(np.abs(slip) + 1e-40))
    #print('deficit')
    #plot_signs(slip_deficit)
    #plot_fields(np.log10(np.abs(slip_deficit) + 1e-40))
    print('slip')
    plot_fields(qdm.m, slip)
    print('V')
    #plot_signs(V)
    plot_fields(qdm.m, np.log10(np.abs(V) + 1e-40))
    print('traction')
    min_trac = 0.9 * np.max(traction)
    max_trac = np.max(traction)
    plot_fields(qdm.m, traction)#, levels = np.linspace(min_trac, max_trac, 20))
    print('state')
    plot_fields(qdm.m, state)


class QDPlotData:
    def __init__(self, data):
        self.data = data
        m, self.qd_cfg, t, y = self.data
        self.qdm = QDMeshData((m.pts, m.tris))
        self.t = np.array(t)
        siay = 365.25 * 24 * 3600
        self.t_years = self.t / siay
        self.y = np.array(y)
        self.slip_to_traction = None

        self.n_steps = self.y.shape[0]
        self.min_state = [0] * self.n_steps
        self.max_V = [0] * self.n_steps
        self.slip = [0] * self.n_steps
        self.state = [0] * self.n_steps
        self.V = [0] * self.n_steps
        self.dt = [0] * self.n_steps
        for i in range(0, self.n_steps):
            self.slip[i], self.state[i] = separate_slip_state(self.y[i])
            self.min_state[i] = np.min(self.state[i])
            if i > 0:
                self.dt[i] = self.t[i] - self.t[i - 1]
                self.V[i] = (self.slip[i] - self.slip[i - 1]) / self.dt[i]
            self.max_V[i] = np.max(self.V[i])

    def setup_operator(self):
        if self.slip_to_traction is None:
            self.slip_to_traction = get_slip_to_traction(qdm, qd_cfg)

    def summary(self):
        print('plotting', np.max(self.t_years), 'years of data')
        plt.figure(figsize = (16,16))
        plt.subplot(221)
        plt.plot(self.t_years[1:], self.min_state[1:])
        plt.xlabel('step')
        plt.ylabel('$\min(\Psi)$')
        plt.subplot(222)
        plt.plot(self.t_years[1:], np.log10(np.abs(self.max_V[1:]) + 1e-40))
        plt.xlabel('step')
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

    def nicefig(self, field, levels, contour_levels, cmap, filepath = None):
        contour_levels = levels[::3]

        pt_field = np.empty(self.qdm.m.pts.shape[0])
        pt_field[self.qdm.m.tris] = field.reshape((-1,3))

        fig = plt.figure(figsize = (10,8))
        ax = plt.gca()
        div = make_axes_locatable(ax)
        cax = div.append_axes('right', '5%', '5%')

        cntf = ax.tricontourf(
            self.qdm.m.pts[:,0], self.qdm.m.pts[:,2], self.qdm.m.tris, pt_field,
            cmap = cmap, levels = levels, extend = 'both'
        )
        ax.tricontour(
            self.qdm.m.pts[:,0], self.qdm.m.pts[:,2], self.qdm.m.tris, pt_field,
            levels = contour_levels, extend = 'both', linestyles = 'solid',
            linewidths = 0.75, colors = ['#FFFFFF'] * contour_levels.shape[0]
        )
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim([-8000, 8000])
        ax.set_ylim([-8000, 8000])
        #ax.text(-8000, 8050, '%.9f' % t_years[i])

        fig.colorbar(cntf, cax = cax)
        if filepath is None:
            plt.show()
        else:
            plt.savefig(filepath, bbox_inches = 'tight', dpi = 300)
        plt.close()


    def qd_video(self, steps_to_plot, field_data_fnc, video_name = None, make_gif = False):

        def get_frame_name(frame_idx, n_frames):
            digits = len(str(n_frames))
            return '%0*d' % (digits, frame_idx)

        if video_name is not None:
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
            filepath = None if video_name is None else f'{video_name}/{frame_name}.png'
            self.nicefig(*field_data, filepath = filepath)
        if make_gif:
            os.system(f'convert -delay 10 -loop 0 {video_name}/*.png {video_name}.gif')

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
