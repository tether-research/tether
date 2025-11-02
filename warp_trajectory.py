import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.interpolate import interp1d 
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from utils.geometry_utils import compose_transformation_matrix, add_poses
from utils.misc_utils import load_trajectory, filter_close_indices


def warp_trajectory(cfg, demo_dir, output_dir, show=False):
    trajectory = load_trajectory(demo_dir / "trajectory_demo.npy")
    keypoint_indices = np.load(demo_dir / "gripper_keypoints.npy")

    with open(demo_dir / "pipeline" / "correspondence" / f"warp_response.json", "r") as f:
        keypoint_deltas_json = json.load(f)
        keypoint_deltas_json = {int(k): v for k, v in keypoint_deltas_json.items()}

    for i in range(len(keypoint_indices)):
        T_delta = compose_transformation_matrix(
            np.array(keypoint_deltas_json[i]["position_delta"]),
            np.array(keypoint_deltas_json[i]["orientation_delta"])
        )
        keypoint_deltas_json[i]["transformation"] = T_delta
    
    # Warp segments between consecutive keypoints
    # Keep initial position constant, warp final position relative to last warp
    warped_trajectory = []
    infos = {}
    for i in range(len(keypoint_deltas_json)+1):
        if i == 0:
            t_start, t_end = 0, keypoint_indices[i]
            T_start, T_end = np.eye(4), keypoint_deltas_json[i]["transformation"]
        elif i == len(keypoint_deltas_json):
            t_start, t_end = keypoint_indices[i-1], len(trajectory) - 1
            T_start, T_end = keypoint_deltas_json[i-1]["transformation"], keypoint_deltas_json[i-1]["transformation"]
        else:
            t_start, t_end = keypoint_indices[i-1], keypoint_indices[i]
            T_start, T_end = keypoint_deltas_json[i-1]["transformation"], keypoint_deltas_json[i]["transformation"]
        segment, segment_infos = warp_segment(trajectory[t_start:t_end + 1], T_start, T_end)
        warped_trajectory.append(segment)
        infos[str(i)] = segment_infos
    warped_trajectory = np.concatenate(warped_trajectory, axis=0)

    np.save(output_dir / f"trajectory_warp.npy", warped_trajectory)
    with open(output_dir / f"trajectory_warp_infos.json", "w") as f:
        json.dump(infos, f, indent=4)
    if show:
        plot_gradient_trajectories(trajectory, warped_trajectory, keypoint_indices)
    return warped_trajectory, infos


def warp_segment(segment, T_start, T_end, normalize_velocity=True):
    t1, t2 = T_start[:3, 3], T_end[:3, 3]
    r1, r2 = T_start[:3, :3], T_end[:3, :3]

    P = segment[:, :3]
    A = segment[0, :3]
    B = segment[-1, :3]
    AB = B - A
    AP = P - A
    proj_scale = np.dot(AP, AB) / np.dot(AB, AB)
    proj_points = A + proj_scale[:, None] * AB
    dist_to_start = np.linalg.norm(proj_points - A, axis=1)[:, None]
    dist_to_end = np.linalg.norm(proj_points - B, axis=1)[:, None]
    dist_to_both = dist_to_start + dist_to_end
    t_interp = dist_to_start / dist_to_both * t2 + dist_to_end / dist_to_both * t1
    r_interp = Slerp([0, 1], R.from_matrix([r1, r2]))(np.linspace(0, 1, len(segment))).as_euler("xyz", degrees=False)
    
    warped_segment = np.zeros_like(segment)
    warped_segment[:, :6] = add_poses(np.concatenate([t_interp, r_interp], axis=-1), segment[:, :6])
    warped_segment[:, 6] = segment[:, 6]

    if normalize_velocity:
        return normalize_warping_velocity(segment, warped_segment)
    return warped_segment, {}


def normalize_warping_velocity(demo_trajectory, warped_trajectory):
    assert demo_trajectory.shape[1] == 7
    assert warped_trajectory.shape[1] == 7
    
    demo_traj_pos, warp_traj_pos = demo_trajectory[:, :3], warped_trajectory[:, :3]
    total_dist_demo = np.sum(np.linalg.norm(np.diff(demo_traj_pos, axis=0), axis=1))
    total_dist_warp = np.sum(np.linalg.norm(np.diff(warp_traj_pos, axis=0), axis=1))
    dist_scale = total_dist_warp / total_dist_demo
    new_num_points = int(round(dist_scale * len(demo_trajectory)))

    idx_to_xyz = linear_interpolate_xyz(warp_traj_pos)
    idx_to_rpy = spherical_interpolate_rpy(warped_trajectory[:, 3:6])

    normalized_warped_trajectory = np.zeros((new_num_points, 7))
    for i, t in enumerate(np.linspace(0, 1, new_num_points)):
        normalized_warped_trajectory[i, :3] = idx_to_xyz(t)
        normalized_warped_trajectory[i, 3:6] = idx_to_rpy(t)
    normalized_warped_trajectory[:, 6] = discretize_gripper(demo_trajectory[:, 6], new_num_points)

    infos = {
        "dist_scale": float(dist_scale),
        "total_dist_demo": float(total_dist_demo),
        "total_dist_warp": float(total_dist_warp),
        "new_num_points": int(new_num_points),
    }
    return normalized_warped_trajectory, infos


def linear_interpolate_xyz(trajectory_xyz):
    trajectory_xyz = np.atleast_1d(trajectory_xyz)
    if trajectory_xyz.ndim == 1:
        trajectory_xyz = trajectory_xyz[:, None]
    deltas = np.diff(trajectory_xyz, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    cumulative_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])
    cumulative_lengths += np.arange(len(cumulative_lengths)) * 1e-10
    progress = cumulative_lengths / cumulative_lengths[-1]

    indices = np.linspace(0, 1, len(trajectory_xyz))
    index_to_progress = interp1d(indices, progress, kind='linear', fill_value="extrapolate")
    progress_to_xyz = [
        interp1d(progress, trajectory_xyz[:, i], kind='linear', bounds_error=False, fill_value="extrapolate")
        for i in range(trajectory_xyz.shape[1])
    ]
    def index_to_xyz(idx):
        prog = np.clip(np.atleast_1d(index_to_progress(idx)), 0.0, 1.0)
        if prog.ndim == 1:
            prog = prog[:, None]
        return np.hstack([f(prog[:, 0]).reshape(-1, 1) for f in progress_to_xyz])
    return index_to_xyz


def spherical_interpolate_rpy(trajectory_rpy):
    def quaternion_angular_dist(q1, q2):
        dot = np.sum(q1 * q2, axis=-1)
        dot = np.clip(dot, -1.0, 1.0)
        return 2 * np.arccos(np.abs(dot))
    quaternions = R.from_euler('xyz', trajectory_rpy).as_quat()
    arc_lengths = quaternion_angular_dist(quaternions[:-1], quaternions[1:])
    cumulative_lengths = np.concatenate([[0], np.cumsum(arc_lengths)])
    cumulative_lengths += np.arange(len(cumulative_lengths)) * 1e-10
    progress = cumulative_lengths / cumulative_lengths[-1]

    indices = np.linspace(0, 1, len(trajectory_rpy))
    index_to_progress = interp1d(indices, progress, kind='linear', fill_value="extrapolate")
    progress_to_rot = Slerp(progress, R.from_quat(quaternions))
    def index_to_rpy(idx):
        prog = np.clip(np.atleast_1d(index_to_progress(idx)), 0.0, 1.0)
        return progress_to_rot(prog).as_euler('xyz')
    return index_to_rpy


def find_gripper_change_indices(gripper_values):
    diff = np.append(np.diff(gripper_values), 0)
    diff_threshold=0.01
    close_indices = np.where(
        (diff[1:] > diff_threshold) &  # increasing
        (np.abs(diff[:-1]) <= diff_threshold)  # previous step is near zero slope
    )[0] + 1
    close_indices = filter_close_indices(close_indices, closeness_threshold=10)
    open_indices = np.where(
        (diff[1:] < -diff_threshold) &  # decreasing
        (np.abs(diff[:-1]) <= diff_threshold)  # previous step is near zero slope
    )[0] + 1
    open_indices = filter_close_indices(open_indices, closeness_threshold=10)
    return close_indices, open_indices


def discretize_gripper(demo_gripper, new_length):
    N = len(demo_gripper)
    new_gripper = np.zeros(new_length, dtype=demo_gripper.dtype)

    close_indices, open_indices = find_gripper_change_indices(demo_gripper)
    assert len(close_indices) <= 1 and len(open_indices) <= 1, "There should be at most one close and one open index."
    if len(open_indices) == 1:
        open_idx = open_indices[0] + 1
        if open_idx < (N - open_idx):
            open_indices[0] = open_idx
        else:
            open_indices[0] = new_length - (N - open_idx)
    if len(close_indices) == 1:
        close_idx = close_indices[0] + 1
        if close_idx < (N - close_idx):
            close_indices[0] = close_idx
        else:
            close_indices[0] = new_length - (N - close_idx)

    if np.all(demo_gripper > 0):
        new_gripper[:] = 1
    elif len(close_indices) == 0 and len(open_indices) == 0:
        new_gripper = np.all(demo_gripper > 0)
    elif len(close_indices) == 0 and len(open_indices) == 1:
        new_gripper[:open_indices[0]] = 1
        new_gripper[open_indices[0]:] = 0
    elif len(close_indices) == 1 and len(open_indices) == 0:
        new_gripper[close_indices[0]:] = 1
        new_gripper[:close_indices[0]] = 0
    else:
        assert close_indices[0] < open_indices[0], "The close index should be before the open index."
        new_gripper[:close_indices[0]] = 0
        new_gripper[close_indices[0]:open_indices[0]] = 1
        new_gripper[open_indices[0]:] = 0
    return new_gripper


def plot_gradient_trajectory(trajectory, ax, cmap='viridis', label='Trajectory'):
    n_points = trajectory.shape[0]
    time_array = np.arange(n_points)
    trajectory = trajectory[:, :3]
    points = trajectory.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = Line3DCollection(segments, cmap=cmap)
    lc.set_array(time_array[:-1])
    lc.set_linewidth(2)
    ax.add_collection(lc)
    ax.text2D(0.05, 0.95, label, transform=ax.transAxes)
    return lc


def plot_gradient_trajectories(trajectory, warped_trajectory, keypoint_indices):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    lc1 = plot_gradient_trajectory(trajectory, ax, cmap='viridis', label='Original Trajectory')
    lc2 = plot_gradient_trajectory(warped_trajectory, ax, cmap='rainbow', label='Warped Trajectory')
    all_points = np.concatenate((trajectory, warped_trajectory), axis=0)
    x_vals, y_vals, z_vals = all_points[:, 0], all_points[:, 1], all_points[:, 2]
    ax.auto_scale_xyz(x_vals, y_vals, z_vals)
    cbar1 = plt.colorbar(lc1, ax=ax, shrink=0.5, pad=0.1)
    cbar1.set_label('Original Time')
    cbar2 = plt.colorbar(lc2, ax=ax, shrink=0.5, pad=0.15)
    cbar2.set_label('Warped Time')
    line1 = mlines.Line2D([], [], color='black', label='Original Trajectory')
    ax.scatter(trajectory[keypoint_indices, 0],
               trajectory[keypoint_indices, 1],
               trajectory[keypoint_indices, 2],
               c='r', label="Original Keypoints")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.tight_layout()
    plt.show()


def plot_trajectory_warping(original_trajectory, warped_trajectory, keypoint_indices):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(original_trajectory[:, 0], original_trajectory[:, 1], original_trajectory[:, 2], label="Original trajectory")
    ax.plot(warped_trajectory[:, 0], warped_trajectory[:, 1], warped_trajectory[:, 2], label="Warped trajectory")
    ax.scatter(original_trajectory[keypoint_indices, 0],
               original_trajectory[keypoint_indices, 1],
               original_trajectory[keypoint_indices, 2],
               c='r', label="Original Keypoints")
    ax.scatter(warped_trajectory[keypoint_indices, 0],
               warped_trajectory[keypoint_indices, 1],
               warped_trajectory[keypoint_indices, 2],
               c='g', label="Warped Keypoints")
    for i in range(len(original_trajectory)):
        ax.text(original_trajectory[i, 0], original_trajectory[i, 1], original_trajectory[i, 2], str(i))
    for i in range(len(warped_trajectory)):
        ax.text(warped_trajectory[i, 0], warped_trajectory[i, 1], warped_trajectory[i, 2], str(i))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

def plot_trajectories(trajectory_path1, trajectory_path2):
    trajectory1 = load_trajectory(trajectory_path1)
    trajectory2 = load_trajectory(trajectory_path2)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory1[:, 0], trajectory1[:, 1], trajectory1[:, 2], label="Trajectory 1")
    ax.plot(trajectory2[:, 0], trajectory2[:, 1], trajectory2[:, 2], label="Trajectory 2")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

