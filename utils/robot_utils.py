import shutil
import subprocess
from pathlib import Path
import zerorpc

from utils.timer_utils import timer


"""
NOTE: Replace the code here with your own robot infra to collect images and run trajectories.
"""


remote_runner = zerorpc.Client(heartbeat=None, timeout=None)
remote_runner.connect("tcp://172.16.0.1:4545")
print("Remote runner connected to tcp://172.16.0.1:4545")


@timer
def collect_scene_image(cfg):
    image_dir = remote_runner.save_camera_feed()
    subprocess.run(["rsync", "-avz", f"eth.franka.franka-laptop:{image_dir}", f"{cfg.scene_path}"], check=True, stdout=subprocess.DEVNULL)
    scene_name = Path(image_dir).name
    if not (cfg.scene_path / scene_name).exists():
        print(f"Could not find scene {scene_name}!")
        return None
    for camera_name, camera_id in cfg.setting.cameras.items():
        if not (cfg.scene_path / scene_name / f"{camera_id}_left.jpg").exists():
            print(f"Could not find camera {camera_name} in scene {scene_name}!")
            return None
        shutil.copyfile(cfg.scene_path / scene_name / f"{camera_id}_left.jpg", cfg.scene_path / scene_name / f"{camera_name}.jpg")
    return scene_name


@timer
def send_trajectory(cfg, demo_dir, task, save=True):
    print("Sending trajectory...")
    subprocess.run(["rsync", "-avz", f"{demo_dir}/pipeline/trajectory_eva.npy", "eth.franka.franka-laptop:/home/franka/eva/base_trajectories/default/trajectory.npy"], check=True, stdout=subprocess.DEVNULL)
    rollout_dir = remote_runner.run_trajectory_warping("default", "default", "off", task)
    if save:
        subprocess.run(["rsync", "-avz", "--exclude='*.svo2'", "--exclude='*.jpg'", f"eth.franka.franka-laptop:{rollout_dir}", f"{cfg.rollout_path}"], check=True, stdout=subprocess.DEVNULL)
    return Path(rollout_dir)

