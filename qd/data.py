import os

import numpy as np

"""
Don't save anything!
"""
class NullData:
    def initialized(self, _):
        pass

    def stepped(self, _):
        pass

"""
Do not use. Exists for compatibility with old data.
"""
class MonolithicDataSaver:
    def __init__(self, save_freq = 500, filename = 'results.npy'):
        self.filename = filename

    def initialized(self):
        pass

    def stepped(self, integrator):
        if integrator.step_idx() == self.save_freq:
            np.save(
                self.filename,
                np.array([integrator.model.m, qd_cfg, h_t, h_y], dtype = np.object)
            )

def initial_data_path(folder_name):
    return os.path.join(folder_name, 'initial_data.npy')

"""
Saves chunks of time steps in separate files for easy reloading.
"""
class ChunkedDataSaver:
    def __init__(self, chunk_size = 100, folder_name = 'data'):
        self.chunk_size = chunk_size
        self.folder_name = folder_name
        self.check_folder()

    def check_folder(self):
        if os.path.exists(self.folder_name):
            raise Exception(f'Data folder \'{self.folder_name}\' already exists. Please move or delete.')
        else:
            os.makedirs(self.folder_name)

    def initialized(self, integrator):
        np.save(
            initial_data_path(self.folder_name),
            np.array([
                integrator.model.m, integrator.model.cfg,
                integrator.init_conditions
            ], dtype = np.object)
        )

    def stepped(self, integrator):
        step_idx = integrator.step_idx()
        if step_idx % self.chunk_size == 0:
            step_data_path = os.path.join(self.folder_name, f'{step_idx}.npy')
            np.save(
                step_data_path,
                np.array([
                    integrator.h_t[-self.chunk_size:],
                    integrator.h_y[-self.chunk_size:]
                ], dtype = np.object)
            )

class ChunkedDataLoader:
    def __init__(self, folder_name = 'data'):
        self.folder_name = folder_name
        self.idxs = []
        self.ts = None
        self.ys = None

        self.load_initial_data()
        self.load_new_files()

    def load_initial_data(self):
        self.m, self.cfg, self.init_conditions = np.load(
            initial_data_path(self.folder_name)
        )

    def load_new_files(self):
        new_idxs = []
        for f in os.listdir(self.folder_name):
            if not os.path.isfile(os.path.join(self.folder_name, f)):
                continue

            base_name, ext = os.path.splitext(f)
            if not base_name.isdecimal():
                continue
            new_idxs.append(int(base_name))
        new_idxs.sort()

        new_ts = np.empty(new_idxs[-1])
        new_ys = np.empty((new_idxs[-1], self.m.tris.shape[0] * 12))

        if self.ts is not None:
            new_ts[:self.idxs[-1]] = self.ts
            new_ys[:self.idxs[-1]] = self.ys

        self.ts = new_ts
        self.ys = new_ys

        for i in new_idxs:
            chunk = np.load(os.path.join(self.folder_name, f'{i}.npy'))
            n_steps = len(chunk[0])
            self.ts[i - n_steps:i] = chunk[0]
            for j in range(n_steps):
                self.ys[i - n_steps + j] = chunk[1][j]

        self.idxs += new_idxs

def load(folder_name = 'data'):
    return ChunkedDataLoader()
