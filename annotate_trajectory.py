import numpy as np
import cv2
from PIL import Image
import scipy
import scipy.interpolate
import imageio
import hydra

from utils.calibration_utils import load_camera_extrinsics, load_camera_intrinsics
from utils.geometry_utils import transform_trajectory_world_to_camera
from utils.annotation_utils import create_trajectory_overlay, create_keypoint_overlays, create_gripper_overlays, create_label_overlay 
from utils.misc_utils import load_trajectory


def annotate_trajectory(
    cfg,
    trajectory_path,
    keypoint_path,
    media_path,
    extrinsics_path,
    intrinsics_path,
    camera_id,
    output_path,
    densify_factor=64,
    align_action=False,
):
    if not media_path.exists():
        print(f"Media path {media_path} does not exist for annotation!")
        return

    camera_extrinsics = load_camera_extrinsics(extrinsics_path, camera_id)
    camera_intrinsics = load_camera_intrinsics(intrinsics_path, camera_id)
    
    trajectory = load_trajectory(trajectory_path)
    camera_frame_trajectory = transform_trajectory_world_to_camera(trajectory, camera_extrinsics)

    trajectory_overlay_animator, keypoint_overlay_animator, gripper_overlays = None, None, None
    _, trajectory_overlay_animator = create_trajectory_overlay(camera_frame_trajectory, camera_intrinsics, gripper_len=cfg.setting.gripper_len, densify_factor=densify_factor)
    if keypoint_path is not None:
        keypoint_indices = np.load(keypoint_path)
        _, _, keypoint_overlay_animator = create_keypoint_overlays(camera_frame_trajectory, keypoint_indices, camera_intrinsics, gripper_len=cfg.setting.gripper_len)
    gripper_overlays = create_gripper_overlays(camera_frame_trajectory, camera_intrinsics, gripper_len=cfg.setting.gripper_len)

    media_mode = "video" if media_path.suffix == ".mp4" else "image"
    media_source = cv2.VideoCapture(media_path) if media_mode == "video" else Image.open(media_path)
    video_writer = imageio.get_writer(output_path, fps=cfg.setting.control_hz)

    if media_mode == "video" and align_action:
        # Used to skip the first frame of the rollout video to align with the action command
        # This is used when the trajectory is the action given to the robot instead of the state extracted from the robot
        ret, frame = media_source.read()
        if not ret:
            print(f"Failed to read video from {media_path}!")
            return
        video_writer.append_data(np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    for i in range(len(trajectory)):
        if media_mode == "video":
            ret, frame = media_source.read()
            if not ret:
                break
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            frame = media_source.copy()
        trajectory_overlay = next(trajectory_overlay_animator)
        frame.paste(trajectory_overlay, (0, 0), trajectory_overlay)
        frame.paste(gripper_overlays[i], (0, 0), gripper_overlays[i])
        if keypoint_overlay_animator is not None:
            keypoint_overlay = next(keypoint_overlay_animator)
            frame.paste(keypoint_overlay, (0, 0), keypoint_overlay)
        video_writer.append_data(np.array(frame))
    video_writer.close()
    if media_mode == "video":
        media_source.release()


def annotate_demo_trajectory(cfg, demo_dir, output_dir, densify_factor=64):
    output_dir = output_dir / "annotations"
    output_dir.mkdir(exist_ok=True)
    
    for camera_name, camera_id in cfg.setting.cameras.items():
        annotate_trajectory(
            cfg,
            trajectory_path=demo_dir / "trajectory.npz",
            keypoint_path=demo_dir / "gripper_keypoints.npy",
            media_path=demo_dir / "recordings" / f"{camera_name}.mp4",
            extrinsics_path=demo_dir,
            intrinsics_path=demo_dir,
            camera_id=camera_id,
            output_path=output_dir / f"trajectory_{camera_name}.mp4",
            densify_factor=densify_factor
        )


def annotate_rollout_trajectory(cfg, rollout_dir, demo_dir, output_dir, densify_factor=64):
    output_dir = output_dir / "annotations"
    output_dir.mkdir(exist_ok=True)
    for camera_name, camera_id in cfg.setting.cameras.items():
        annotate_trajectory(
            cfg,
            trajectory_path=rollout_dir / f"pipeline_{demo_dir.name}" / "trajectory_final.npy",
            keypoint_path=rollout_dir / "gripper_keypoints.npy",
            media_path=rollout_dir / "recordings" / f"{camera_name}.mp4",
            extrinsics_path=rollout_dir,
            intrinsics_path=rollout_dir,
            camera_id=camera_id,
            output_path=output_dir / f"trajectory_{camera_name}.mp4",
            densify_factor=densify_factor,
            align_action=True
        )

