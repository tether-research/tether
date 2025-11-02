import json
import numpy as np

from utils.geometry_utils import euler_to_rmat


def load_camera_extrinsics(load_path, camera_id):
    load_path = load_path if load_path.name == "calibration.json" else load_path / "calibration.json"
    with open(load_path, "r") as f:
        calibration_dict = json.load(f)
    camera_extrinsics_vec = np.array(calibration_dict[f"{camera_id}_left"]["extrinsics"])
    camera_extrinsics = np.eye(4)
    camera_extrinsics[:3, :3] = euler_to_rmat(camera_extrinsics_vec[3:])
    camera_extrinsics[:3, 3] = camera_extrinsics_vec[:3]
    camera_extrinsics = np.linalg.inv(camera_extrinsics)
    return camera_extrinsics


def load_camera_intrinsics(load_path, camera_id):
    load_path = load_path if load_path.name == "calibration.json" else load_path / "calibration.json"
    with open(load_path, "r") as f:
        calibration_dict = json.load(f)
    camera_intrinsics = np.array(calibration_dict[f"{camera_id}_left"]["intrinsics"])
    return camera_intrinsics

