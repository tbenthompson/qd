import numpy as np
import subprocess
import os
import uuid
import cloudpickle
import multiprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

LINE_WIDTH = 0.25
SIAY = 60 * 60 * 24 * 365.25
USE_N_CORES = 30


# Read Ben's data for Cascadia
pts, _tris, t, slip_all, state_all = np.load("data_for_brendan.npy")
dt_vec = np.diff(t)
slip_magnitude_all = np.sqrt(
    slip_all[:, :, 0] ** 2 + slip_all[:, :, 1] ** 2 + slip_all[:, :, 2] ** 2
)
slip_diff_all = np.diff(slip_magnitude_all, axis=0)
slip_rate_all = slip_diff_all / dt_vec[:, np.newaxis]
slip_rate_log_all = np.log10(np.abs(slip_rate_all))


def plotter_executor(data):
    serialized_fnc, frame_idx, filepath = data
    print("frame =", frame_idx)
    fnc = cloudpickle.loads(serialized_fnc)
    fnc(frame_idx, filepath)


def make_video(video_name, n_frames, plotter_fnc, framerate=30):
    os.makedirs(video_name)
    digits = len(str(n_frames))
    pool = multiprocessing.Pool(processes=USE_N_CORES)
    serialized_fnc = cloudpickle.dumps(plotter_fnc)
    job_data = []
    for frame_idx in range(n_frames):
        frame_name = "%0*d" % (digits, frame_idx)
        filepath = f"{video_name}/{frame_name}.png"
        job_data.append((serialized_fnc, frame_idx, filepath))
    pool.map(plotter_executor, job_data)

    cmd = [
        "ffmpeg",
        "-framerate",
        str(framerate),
        "-i",
        f"{video_name}/%0{digits}d.png",
        "-c:v",
        "libx264",
        "-r",
        "30",
        "-y",
        "-v",
        "32",
        video_name + ".mp4",
    ]
    print("running", '"' + " ".join(cmd) + '"')
    for line in execute(cmd):
        print(line, end="")


def execute(cmd):
    popen = subprocess.Popen(cmd, stderr=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stderr.readline, ""):
        yield stdout_line
    popen.stderr.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def f(idx, filepath):
    # idx = idx + 500
    plt.figure(figsize=(6, 6))

    # Show histories for each particle
    for i in range(0, slip_rate_log_all.shape[1]):
        x_history = slip_rate_log_all[0 : idx + 1, i]
        y_history = state_all[0 : idx + 1, i]
        plt.plot(
            x_history,
            y_history,
            "-k",
            linewidth=0.25,
            color=[0.90, 0.90, 0.90],
            alpha=0.1,
        )

    plt.scatter(
        slip_rate_log_all[idx, :],
        state_all[idx, :],
        c=np.abs(pts[:, 2]),
        s=10,
        alpha=1.0,
        edgecolors="k",
        linewidths=LINE_WIDTH,
        zorder=30,
        cmap=plt.get_cmap("plasma"),
    )

    tt = t[idx] / t[-1] * 14
    x_fill = np.array([-14, -14 + tt, -14 + tt, -14])
    y_fill = np.array([0.79, 0.79, 0.8, 0.8])
    plt.fill(x_fill, y_fill, "grey")
    plt.xlabel("log(v)")
    plt.ylabel("state")
    plt.xlim([-14, 0])
    plt.ylim([0.4, 0.8])
    plt.xticks([-14, -7, 0])
    plt.yticks([0.4, 0.6, 0.8])
    plt.title("t = " + "{:010.9}".format(t[idx] / SIAY), fontsize=10)
    plt.savefig(filepath, dpi=300)


def main():
    make_video("vid_" + uuid.uuid4().hex, 1998, f)


if __name__ == "__main__":
    main()
