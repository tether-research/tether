import numpy as np

from utils.misc_utils import load_trajectory, filter_close_indices


def douglas_peucker(points, prior_indices=None, epsilon=0.1):
    """
    In the final version of Tether, we simply use gripper open/close as keypoints.
    However, this function may be useful for future work.
    """

    def recurse(points, indices, epsilon):
        start, end = points[0], points[-1]
        line_vec = end - start
        line_len = np.linalg.norm(line_vec) line_dir = line_vec / line_len vec_to_points = points - start t = np.dot(vec_to_points, line_dir)
        t_clamped = np.clip(t, 0, line_len)
        projection = np.outer(t_clamped, line_dir) + start
        distances = np.linalg.norm(points - projection, axis=1)

        max_dist_idx = np.argmax(distances)
        max_dist = distances[max_dist_idx]

        if max_dist > epsilon:
            left_points, left_indices = recurse(points[:max_dist_idx + 1], indices[:max_dist_idx + 1], epsilon)
            right_points, right_indices = recurse(points[max_dist_idx:], indices[max_dist_idx:], epsilon)
            return np.vstack((left_points[:-1], right_points)), np.concatenate((left_indices[:-1], right_indices))
        else:
            return np.vstack((start, end)), np.array([indices[0], indices[-1]])

    indices = np.arange(len(points))
    if prior_indices is None or len(prior_indices) == 0:
        selected_points, selected_indices = recurse(points, indices, epsilon)
    else:
        prior_indices = np.sort(prior_indices)
        if prior_indices[0] != 0:
            prior_indices = np.concatenate(([0], prior_indices))
        if prior_indices[-1] != len(points) - 1:
            prior_indices = np.concatenate((prior_indices, [len(points) - 1]))
        selected_points, selected_indices = [], []
        for i in range(len(prior_indices) - 1):
            start, end = prior_indices[i], prior_indices[i + 1]
            cur_selected_points, cur_selected_indices = recurse(points[start:end + 1], indices[start:end + 1], epsilon)
            if i > 0:
                cur_selected_points, cur_selected_indices = cur_selected_points[1:], cur_selected_indices[1:]
            selected_points.append(cur_selected_points)
            selected_indices.append(cur_selected_indices)
        selected_points = np.vstack(selected_points)
        selected_indices = np.concatenate(selected_indices)
    return selected_points, selected_indices


def extract_keypoint_trajectory(cfg, demo_dir, output_dir):
    trajectory = load_trajectory(demo_dir / "trajectory_demo.npy")
    trajectory_pos, trajectory_gripper = trajectory[:, :3], trajectory[:, -1]

    diff = np.append(np.diff(trajectory_gripper), 0)  # Add 0 to consider last frame
    diff_threshold=0.01
    close_indices = np.where(
        (diff[:-1] > diff_threshold) &  # increasing
        (np.abs(diff[1:]) <= diff_threshold)  # next step is near zero slope
    )[0] + 1
    close_indices = filter_close_indices(close_indices, closeness_threshold=10)
    open_indices = np.where(
        (diff[1:] < -diff_threshold) &  # decreasing
        (np.abs(diff[:-1]) <= diff_threshold)  # previous step is near zero slope
    )[0] + 1
    open_indices = filter_close_indices(open_indices, closeness_threshold=10)
    gripper_keypoint_indices = np.sort(np.concatenate((close_indices, open_indices)))
    keypoint_indices = gripper_keypoint_indices.copy()

    _, keypoint_indices = douglas_peucker(trajectory_pos, prior_indices=keypoint_indices)

    np.save(output_dir / f"keypoints.npy", keypoint_indices)
    np.save(output_dir / f"gripper_keypoints.npy", gripper_keypoint_indices)
    return keypoint_indices, open_indices, close_indices

