
import numpy as np
import subprocess
import os
import shutil
import json
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from datetime import datetime

from utils.geometry_utils import euler_to_rmat


def load_trajectory(path):
    if path.suffix == ".npz":
        return np.load(path)["states"][:, :7]
    else:
        return np.load(path, allow_pickle=True)


def modify_timestep_position(timestep, direction, magnitude, offset_vector=[0, 0, 1]):
    offset_vector = np.array(offset_vector)
    pos, rot, gripper = timestep[:3], timestep[3:6], timestep[6:]
    new_pos = pos + direction * euler_to_rmat(rot) @ offset_vector * magnitude
    return new_pos


def modify_trajectory_position(trajectory, direction, magnitude):
    # For each timestep, moves the position along the orientation either forward or backward along the gripper
    # We do this because we want the editing trajectory to represent the gripper fingers, not the gripper base
    # direction: 1 for forward (base to fingers), -1 for backward (fingers to base)

    if len(trajectory.shape) == 1:
        return modify_trajectory_position(trajectory[None, :], direction, magnitude)[0]
    
    new_trajectory = np.zeros_like(trajectory)
    for i in range(len(trajectory)):
        new_trajectory[i, :3] = modify_timestep_position(trajectory[i], direction, magnitude)
        new_trajectory[i, 3:6] = trajectory[i, 3:6]
        new_trajectory[i, 6:] = trajectory[i, 6:]
    return new_trajectory


def add_trajectory_prefix(trajectory):
    start_pos, start_rot = np.array([0.05, 0, 0.6]), np.array([3.14, 0.11, 0])

    magnitude = np.linalg.norm(trajectory[0, :3] - start_pos)
    num_steps = int(np.ceil(magnitude / 0.02))
    if num_steps == 0:
        return trajectory
    new_trajectory = np.zeros((num_steps + len(trajectory), trajectory.shape[1]))
    new_trajectory[:num_steps, :3] = np.linspace(start_pos, trajectory[0, :3], num_steps)
    new_trajectory[:num_steps, 3:6] = Slerp([0, 1], R.from_euler("xyz", [start_rot, trajectory[0, 3:6]], degrees=False))(np.linspace(0, 1, num_steps)).as_euler("xyz", degrees=False)
    new_trajectory[:num_steps, 6:] = trajectory[0, 6:]
    new_trajectory[num_steps:, :] = trajectory
    return new_trajectory, num_steps


def prepare_trajectory(cfg, demo_dir, direction, output_dir):
    assert direction in [1, -1]
    if direction == 1:
        trajectory = load_trajectory(demo_dir / "trajectory.npz")
        trajectory = modify_trajectory_position(trajectory, direction, magnitude=cfg.setting.gripper_len)
        np.save(output_dir / "trajectory_demo.npy", trajectory)
        return None
    elif direction == -1:
        trajectory = load_trajectory(demo_dir / "pipeline" / "trajectory_warp.npy")
        trajectory = modify_trajectory_position(trajectory, direction, magnitude=cfg.setting.gripper_len)
        np.save(output_dir / "pipeline" / "trajectory_final.npy", trajectory)


def format_video(video_path):
    subprocess.run(["ffmpeg", "-i", video_path, "-c:v", "libx264", "-pix_fmt", "yuv420p", video_path.replace(".mp4", "_h264.mp4")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.replace(video_path.replace(".mp4", "_h264.mp4"), video_path)


def slowdown_video(video_path, slowdown_factor=2.0):
    new_path = video_path.replace(".mp4", "_h264_slow.mp4")
    subprocess.run(["ffmpeg", "-i", video_path, "-filter:v", f"setpts={slowdown_factor}*PTS", "-c:v", "libx264", "-pix_fmt", "yuv420p", new_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.replace(new_path, video_path)


def timestamp_to_seconds(timestamp):
    mm, ss = map(int, timestamp.split(":"))
    return mm * 60 + ss


def seconds_to_timestamp(seconds):
    mm = seconds // 60
    ss = seconds % 60
    return f"{mm:02d}:{ss:02d}"


def get_ordered_gpus():
    sp = subprocess.Popen(['gpustat', '--json'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str, _ = sp.communicate()
    gpustats = json.loads(out_str.decode('utf-8'))
    # order by memory usage, lowest first
    gpus = sorted(gpustats['gpus'], key=lambda x: x['memory.used'])
    return [f"cuda:{gpu['index']}" for gpu in gpus]


def get_root_demo_name(demo_name):
    return "_".join(demo_name.split("_")[:-2])


def check_pos_oob(point, oob_bounds):
    x, y, z = point
    if x < oob_bounds.x[0] or x > oob_bounds.x[1]:
        return True
    if y < oob_bounds.y[0] or y > oob_bounds.y[1]:
        return True
    if z < oob_bounds.z[0] or z > oob_bounds.z[1]:
        return True
    return False


def copy_latest_file_from_folder(src, dst):
    if not src.is_dir():
        raise ValueError(f"Source path {src} is not a directory.")
    if not dst.exists():
        dst.mkdir(parents=True)

    folders = [f for f in src.iterdir() if f.is_dir()]
    if not folders:
        raise FileNotFoundError("No folders found in the source directory.")
    latest_folder = max(folders, key=lambda f: f.stat().st_mtime)
    target_path = dst / latest_folder.name
    shutil.copytree(latest_folder, target_path)


def is_valid_rollout(rollout_dir):
    trajectory_file = rollout_dir / "trajectory.npz"
    if not trajectory_file.exists():
        print(f"Skipping {rollout_dir} as it does not contain trajectory.npz")
        return False
    metadata_file = rollout_dir / "metadata.json"
    if not metadata_file.exists():
        print(f"Skipping {rollout_dir} as it does not contain metadata.json")
        return False
    pipeline_dirs = list(rollout_dir.glob("pipeline_*"))
    if len(pipeline_dirs) != 1:
        print(f"Skipping {rollout_dir} as it does not contain exactly one pipeline directory")
        return False
    return True


def get_timestamp():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return timestamp


def filter_close_indices(indices, closeness_threshold=15):
    if len(indices) == 0:
        return indices
    sorted_indices = np.sort(indices)
    filtered = [sorted_indices[0]]
    for i in range(1, len(sorted_indices)):
        if (sorted_indices[i] - filtered[-1]) >= closeness_threshold:
            filtered.append(sorted_indices[i])
    return np.array(filtered)
