import os
from pathlib import Path
from PIL import Image
import numpy as np
import json
from multiprocessing.managers import BaseManager
import shutil

from utils.misc_utils import load_trajectory
from utils.calibration_utils import load_camera_extrinsics, load_camera_intrinsics
from utils.geometry_utils import compute_stereo_triangulation, project_world_coord_to_image, distance_point_to_pixel_ray
from utils.annotation_utils import concatenate_images, create_epipolar_line_overlay, create_point_overlay
from utils.timer_utils import timer
from utils.misc_utils import load_trajectory, modify_timestep_position, check_pos_oob


class GeoAwareManager(BaseManager):
    pass

GeoAwareManager.register("GeoAware")

def load_geo_aware(port):
    manager = GeoAwareManager(address=("localhost", port), authkey=b"geoaware")
    manager.connect()
    return manager.GeoAware()

geo_aware = load_geo_aware(port=50011)

class Mast3rManager(BaseManager):
    pass

Mast3rManager.register("Mast3r")

def load_mast3r():
    manager = Mast3rManager(address=("localhost", 50022), authkey=b"mast3r")
    manager.connect()
    return manager.Mast3r()

mast3r = load_mast3r()


@timer
def run_correspondence(cfg, keypoint_indices, demo_dir, scene_dir, output_dir):
    print("Running correspondence")
    output_dir = Path(output_dir) / "correspondence"
    os.makedirs(output_dir, exist_ok=True)

    camera_name1, camera_name2 = cfg.setting.cameras.keys()
    assert len(cfg.setting.cameras.keys()) == 2
    camera_extrinsics_demo = {camera: load_camera_extrinsics(demo_dir, cfg.setting.cameras[camera]) for camera in cfg.setting.cameras.keys()}
    camera_intrinsics_demo = {camera: load_camera_intrinsics(demo_dir, cfg.setting.cameras[camera]) for camera in cfg.setting.cameras.keys()}
    camera_extrinsics_scene = {camera: load_camera_extrinsics(scene_dir, cfg.setting.cameras[camera]) for camera in cfg.setting.cameras.keys()}
    camera_intrinsics_scene = {camera: load_camera_intrinsics(scene_dir, cfg.setting.cameras[camera]) for camera in cfg.setting.cameras.keys()}
    trajectory = load_trajectory(demo_dir / "trajectory_demo.npy")
    with open(demo_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    warp_response = {i: {} for i in range(len(keypoint_indices))}

    anchor_point_candidates = {}
    anchor_offset_candidates = cfg.setting.anchor_offsets
    warp_keypoint_indices = list(range(len(keypoint_indices)))

    demo_pos = {}
    for keypoint_idx in warp_keypoint_indices:
        anchor_point_candidates[keypoint_idx] = []
        anchor_point_candidates[keypoint_idx] = []
        demo_pos[keypoint_idx] = [None for _ in range(len(anchor_offset_candidates))]
        for i, offset in enumerate(anchor_offset_candidates):
            anchor_point = {}
            timestep = trajectory[keypoint_indices[keypoint_idx]]
            demo_pos[keypoint_idx][i] = modify_timestep_position(timestep, direction=-1, magnitude=offset)
            for camera in cfg.setting.cameras.keys():
                anchor_point[camera] = tuple(project_world_coord_to_image(demo_pos[keypoint_idx][i], camera_intrinsics_demo[camera], camera_extrinsics_demo[camera]).tolist())
            anchor_point_candidates[keypoint_idx].append(anchor_point)

    anchor_correspondence_candidates = {}
    anchor_correspondence_scores = {}
    for keypoint_idx in warp_keypoint_indices:
        anchor_correspondence_candidates[keypoint_idx] = [{} for _ in range(len(anchor_point_candidates[keypoint_idx]))]
        anchor_correspondence_scores[keypoint_idx] = [{} for _ in range(len(anchor_point_candidates[keypoint_idx]))]
    for camera in cfg.setting.cameras.keys():
        source_image_path = demo_dir / "recordings" / "frames" / camera / f"{0:05d}.jpg"
        target_image_path = scene_dir / f"{camera}.jpg"
        source_cache_hit, target_cache_hit = geo_aware.load_images(
            str(source_image_path), str(target_image_path),
            source_crop=cfg.setting.image_crop[camera], target_crop=cfg.setting.image_crop[camera],
            cache_path=(cfg.cache_path / "geo_aware")
        )
        print(f"Geoaware cache hit (source, target): {source_cache_hit}, {target_cache_hit}")
        for keypoint_idx in warp_keypoint_indices:
            for i, anchor_point in enumerate(anchor_point_candidates[keypoint_idx]):
                (x, y), score = geo_aware.compute_correspondence(anchor_point[camera][0], anchor_point[camera][1])
                anchor_correspondence_candidates[keypoint_idx][i][camera] = (x, y)
                anchor_correspondence_scores[keypoint_idx][i][camera] = score

    demo_crossview_correspondence_candidates = {}
    scene_crossview_correspondence_candidates = {}
    for keypoint_idx in warp_keypoint_indices:
        demo_crossview_correspondence_candidates[keypoint_idx] = [{} for _ in range(len(anchor_point_candidates[keypoint_idx]))]
        scene_crossview_correspondence_candidates[keypoint_idx] = [{} for _ in range(len(anchor_point_candidates[keypoint_idx]))]

    for source_camera, target_camera in [[camera_name1, camera_name2], [camera_name2, camera_name1]]:
        source_image_path, target_image_path = demo_dir / "recordings" / "frames" / source_camera / f"{0:05d}.jpg", demo_dir / "recordings" / "frames" / target_camera / f"{0:05d}.jpg"
        mast3r_cache_hit = mast3r.load_images(str(source_image_path), str(target_image_path), cache_path=(cfg.cache_path / "mast3r"))
        print(f"Mast3r Cache Hit: {mast3r_cache_hit}")
        for keypoint_idx in warp_keypoint_indices:
            for i, anchor_points in enumerate(anchor_point_candidates[keypoint_idx]):
                xy, score = mast3r.compute_correspondence(anchor_points[source_camera][0], anchor_points[source_camera][1], cfg.setting.mast3r_error_thres)
                demo_crossview_correspondence_candidates[keypoint_idx][i][target_camera] = xy

    for source_camera, target_camera in [[camera_name1, camera_name2], [camera_name2, camera_name1]]:
        source_image_path, target_image_path = scene_dir / f"{source_camera}.jpg", scene_dir / f"{target_camera}.jpg"
        mast3r_cache_hit = mast3r.load_images(str(source_image_path), str(target_image_path), cache_path=(cfg.cache_path / "mast3r"))
        print(f"Mast3r Cache Hit: {mast3r_cache_hit}")
        for keypoint_idx in warp_keypoint_indices:
            for i, anchor_correspondence_point in enumerate(anchor_correspondence_candidates[keypoint_idx]):
                xy, score = mast3r.compute_correspondence(anchor_correspondence_point[source_camera][0], anchor_correspondence_point[source_camera][1], cfg.setting.mast3r_error_thres)
                scene_crossview_correspondence_candidates[keypoint_idx][i][target_camera] = xy

    for keypoint_idx in warp_keypoint_indices:
        warp_response[keypoint_idx]["infos"] = {}
        best_i, best_anchor_pos, best_corres_pos, best_score, best_corres_infos = -1, None, None, float("-inf"), None
        for i in range(len(anchor_point_candidates[keypoint_idx])):
            anchor_point = anchor_point_candidates[keypoint_idx][i]
            anchor_correspondence = anchor_correspondence_candidates[keypoint_idx][i]

            try:
                anchor_pos, anchor_dist_err, _ = compute_stereo_triangulation(
                    anchor_point[camera_name1], anchor_point[camera_name2],
                    camera_intrinsics_demo[camera_name1], camera_extrinsics_demo[camera_name1],
                    camera_intrinsics_demo[camera_name2], camera_extrinsics_demo[camera_name2]
                )
                corres_pos, corres_dist_err, corres_infos = compute_stereo_triangulation(
                    anchor_correspondence[camera_name1], anchor_correspondence[camera_name2],
                    camera_intrinsics_scene[camera_name1], camera_extrinsics_scene[camera_name1],
                    camera_intrinsics_scene[camera_name2], camera_extrinsics_scene[camera_name2]
                )

                if scene_crossview_correspondence_candidates[keypoint_idx][i][camera_name1] is None:
                    scene_crossview_corres_dist1 = float("inf")
                else:
                    scene_crossview_corres_dist1, _ = distance_point_to_pixel_ray(
                        corres_pos, scene_crossview_correspondence_candidates[keypoint_idx][i][camera_name1], camera_intrinsics_scene[camera_name1], camera_extrinsics_scene[camera_name1]
                    )
                if demo_crossview_correspondence_candidates[keypoint_idx][i][camera_name1] is None:
                    demo_crossview_corres_dist1 = float("inf")
                else:
                    demo_crossview_corres_dist1, _ = distance_point_to_pixel_ray(
                        demo_pos[keypoint_idx][i], demo_crossview_correspondence_candidates[keypoint_idx][i][camera_name1], camera_intrinsics_demo[camera_name1], camera_extrinsics_demo[camera_name1]
                    )

                if scene_crossview_correspondence_candidates[keypoint_idx][i][camera_name2] is None:
                    scene_crossview_corres_dist2 = float("inf")
                else:
                    scene_crossview_corres_dist2, _ = distance_point_to_pixel_ray(
                        corres_pos, scene_crossview_correspondence_candidates[keypoint_idx][i][camera_name2], camera_intrinsics_scene[camera_name2], camera_extrinsics_scene[camera_name2]
                    )
                if demo_crossview_correspondence_candidates[keypoint_idx][i][camera_name2] is None:
                    demo_crossview_corres_dist2 = float("inf")
                else:
                    demo_crossview_corres_dist2, _ = distance_point_to_pixel_ray(
                        demo_pos[keypoint_idx][i], demo_crossview_correspondence_candidates[keypoint_idx][i][camera_name2], camera_intrinsics_demo[camera_name2], camera_extrinsics_demo[camera_name2]
                    )

                crossview_corres_dist_err1 = abs(demo_crossview_corres_dist1 - scene_crossview_corres_dist1)
                crossview_corres_dist_err2 = abs(demo_crossview_corres_dist2 - scene_crossview_corres_dist2)

                corres_pos_offset = modify_timestep_position(
                    np.concatenate((corres_pos, trajectory[keypoint_indices[keypoint_idx], 3:])),
                    direction=1, magnitude=anchor_offset_candidates[i]
                )
            except Exception as e:
                print(f"Triangulation error for correspondence {i}, keypoint {keypoint_idx}")
                continue
            
            correspondence_score = anchor_correspondence_scores[keypoint_idx][i][camera_name1] + anchor_correspondence_scores[keypoint_idx][i][camera_name2]  # [0, 2]
            distance_score = -corres_dist_err  # [-0.1, 0]
            score = 1.0 * correspondence_score + 10.0 * distance_score

            corres_infos = {k: v.tolist() for k, v in corres_infos.items()}
            warp_response[keypoint_idx]["infos"][i] = {
                "correspondence_score": float(correspondence_score),
                "distance_score": float(distance_score),
                "score": float(score),
                "anchor_pos": anchor_pos.tolist(),
                "corres_pos": corres_pos.tolist(),
                "corres_pos_offset": corres_pos_offset.tolist(),
                "anchor_dist_err": float(anchor_dist_err),
                "corres_dist_err": float(corres_dist_err),
                "corres_infos": corres_infos,
            }

            if cfg.setting.name != "sim":
                warp_response[keypoint_idx]["infos"][i]["crossview_corres_dist_err1"] = float(crossview_corres_dist_err1)
                warp_response[keypoint_idx]["infos"][i]["crossview_corres_dist_err2"] = float(crossview_corres_dist_err2)
                warp_response[keypoint_idx]["infos"][i]["demo_crossview_corres_dist1"] = float(demo_crossview_corres_dist1)
                warp_response[keypoint_idx]["infos"][i]["scene_crossview_corres_dist1"] = float(scene_crossview_corres_dist1)
                warp_response[keypoint_idx]["infos"][i]["demo_crossview_corres_dist2"] = float(demo_crossview_corres_dist2)
                warp_response[keypoint_idx]["infos"][i]["scene_crossview_corres_dist2"] = float(scene_crossview_corres_dist2)

            if anchor_dist_err > 0.1 or corres_dist_err > 0.1:
                warp_response[keypoint_idx]["infos"][i]["error"] = "triangulation_distance_error"
                continue
            
            if crossview_corres_dist_err1 > 0.1 or crossview_corres_dist_err2 > 0.1:
                warp_response[keypoint_idx]["infos"][i]["error"] = "crossview_distance_error"
                continue
            
            if check_pos_oob(corres_pos, cfg.setting.oob_bounds):
                warp_response[keypoint_idx]["infos"][i]["error"] = "out_of_bounds"
                continue
            
            if score > best_score:
                best_i = i
                best_anchor_pos = anchor_pos
                best_corres_pos = corres_pos
                best_score = score
                best_corres_infos = corres_infos

        if best_corres_pos is None:
            print(f"Failed to find correspondence anchor for keypoint {keypoint_idx}")
            print(f"Errors: {', '.join(v['error'] for i, v in warp_response[keypoint_idx]['infos'].items())}")
            with open(output_dir / "warp_response.json", "w") as f:
                json.dump(warp_response, f, indent=4)
            with open(output_dir / "corres_infos.json", "w") as f:
                json.dump({
                    "anchor_point_candidates": anchor_point_candidates,
                    "anchor_correspondence_candidates": anchor_correspondence_candidates,
                    "demo_crossview_correspondence_candidates": demo_crossview_correspondence_candidates,
                    "scene_crossview_correspondence_candidates": scene_crossview_correspondence_candidates,
                }, f, indent=4)
            return None

        # Note: we don't need to undo offset here because they cancel out
        warp_response[keypoint_idx]["position_delta"] = (best_corres_pos - best_anchor_pos).tolist()
        warp_response[keypoint_idx]["orientation_delta"] = [0, 0, 0]
        warp_response[keypoint_idx]["best_candidate"] = best_i
    
    if metadata["articulated"]:
        for i in range(1, len(keypoint_indices)):
            warp_response[i]["position_delta"] = warp_response[warp_keypoint_indices[0]]["position_delta"]
            warp_response[i]["orientation_delta"] = warp_response[warp_keypoint_indices[0]]["orientation_delta"]
            warp_response[i]["infos"] = "articulated"

    with open(output_dir / "warp_response.json", "w") as f:
        json.dump(warp_response, f, indent=4)
    with open(output_dir / "corres_infos.json", "w") as f:
        json.dump({
            "anchor_point_candidates": anchor_point_candidates,
            "anchor_correspondence_candidates": anchor_correspondence_candidates,
            "demo_crossview_correspondence_candidates": demo_crossview_correspondence_candidates,
            "scene_crossview_correspondence_candidates": scene_crossview_correspondence_candidates,
        }, f, indent=4)

    return warp_response


def create_correspondence_visualization(cfg, demo_dir, scene_dir, output_dir):
    output_dir = Path(output_dir) / "correspondence"
    keypoint_indices = np.load(demo_dir / "gripper_keypoints.npy")
    warp_keypoint_indices = list(range(len(keypoint_indices)))
    with open(output_dir / "corres_infos.json", "r") as f:
        corres_infos = json.load(f)
    anchor_point_candidates = {int(k): v for k, v in corres_infos["anchor_point_candidates"].items()}
    anchor_correspondence_candidates = {int(k): v for k, v in corres_infos["anchor_correspondence_candidates"].items()}
    demo_crossview_correspondence_candidates = {int(k): v for k, v in corres_infos["demo_crossview_correspondence_candidates"].items()}
    scene_crossview_correspondence_candidates = {int(k): v for k, v in corres_infos["scene_crossview_correspondence_candidates"].items()}

    for keypoint_idx in warp_keypoint_indices:
        for i in range(len(anchor_point_candidates[keypoint_idx])):
            correspondence_images = []
            for camera in cfg.setting.cameras.keys():
                source_image = Image.open(demo_dir / "recordings" / "frames" / camera / f"{0:05d}.jpg")
                target_image = Image.open(scene_dir / f"{camera}.jpg")
                source_x, source_y = anchor_point_candidates[keypoint_idx][i][camera]
                target_x, target_y = anchor_correspondence_candidates[keypoint_idx][i][camera]
                
                source_overlay = create_point_overlay(source_x, source_y, color=(255, 0, 0, 255))
                target_overlay = create_point_overlay(target_x, target_y, color=(255, 0, 0, 255))

                if demo_crossview_correspondence_candidates[keypoint_idx][i][camera] is not None:
                    crossview_x, crossview_y = demo_crossview_correspondence_candidates[keypoint_idx][i][camera]
                    crossview_overlay = create_point_overlay(crossview_x, crossview_y, color=(0, 0, 255, 255))
                    source_image.paste(crossview_overlay, (0, 0), crossview_overlay)
                if scene_crossview_correspondence_candidates[keypoint_idx][i][camera] is not None:
                    crossview_x, crossview_y = scene_crossview_correspondence_candidates[keypoint_idx][i][camera]
                    crossview_overlay = create_point_overlay(crossview_x, crossview_y, color=(0, 0, 255, 255))
                    target_image.paste(crossview_overlay, (0, 0), crossview_overlay)

                source_image.paste(source_overlay, (0, 0), source_overlay)
                target_image.paste(target_overlay, (0, 0), target_overlay)

                correspondence_images.append(concatenate_images(source_image, target_image, text1="source", text2="target"))
            correspondence_image = concatenate_images(*correspondence_images, orientation="vertical")
            correspondence_image.save(output_dir / f"correspondence_{keypoint_idx}_{i}.jpg")


def create_triangulation_visualization(cfg, demo_dir, scene_dir, output_dir):
    output_dir = Path(output_dir) / "correspondence"
    camera_name1, camera_name2 = cfg.setting.cameras.keys()
    camera_extrinsics_scene = {camera: load_camera_extrinsics(scene_dir, cfg.setting.cameras[camera]) for camera in cfg.setting.cameras.keys()}
    camera_intrinsics_scene = {camera: load_camera_intrinsics(scene_dir, cfg.setting.cameras[camera]) for camera in cfg.setting.cameras.keys()}
    keypoint_indices = np.load(demo_dir / "gripper_keypoints.npy")
    warp_keypoint_indices = list(range(len(keypoint_indices)))
    with open(output_dir / "corres_infos.json", "r") as f:
        corres_infos = json.load(f)
    anchor_point_candidates = {int(k): v for k, v in corres_infos["anchor_point_candidates"].items()}
    anchor_correspondence_candidates = {int(k): v for k, v in corres_infos["anchor_correspondence_candidates"].items()}
    with open(output_dir / "warp_response.json", "r") as f:
        warp_response = json.load(f)
        warp_response = {int(k): v for k, v in warp_response.items()}
        for k in warp_response.keys():
            warp_response[k]["infos"] = {int(i): v for i, v in warp_response[k]["infos"].items()}

    for keypoint_idx in warp_keypoint_indices:
        for i in range(len(anchor_point_candidates[keypoint_idx])):
            try:
                corres_infos = warp_response[keypoint_idx]["infos"][i]["corres_infos"]
                corres_infos = {k: np.array(v) for k, v in corres_infos.items()}
                corres_pos = np.array(warp_response[keypoint_idx]["infos"][i]["corres_pos"])
                corres_pos_offset = np.array(warp_response[keypoint_idx]["infos"][i]["corres_pos_offset"])
                left_image, right_image = Image.open(scene_dir / f"{camera_name1}.jpg"), Image.open(scene_dir / f"{camera_name2}.jpg")
                left_overlay = create_epipolar_line_overlay(corres_infos["cam2_center_in_image1"], corres_infos["direction_in_image1"])
                right_overlay = create_epipolar_line_overlay(corres_infos["cam1_center_in_image2"], corres_infos["direction_in_image2"])
                left_corres_point_overlay = create_point_overlay(*anchor_correspondence_candidates[keypoint_idx][i][camera_name1], color=(0, 0, 255, 255))
                right_corres_point_overlay = create_point_overlay(*anchor_correspondence_candidates[keypoint_idx][i][camera_name2], color=(0, 0, 255, 255))
                left_corres_pos_project = project_world_coord_to_image(corres_pos, camera_intrinsics_scene[camera_name1], camera_extrinsics_scene[camera_name1])
                right_corres_pos_project = project_world_coord_to_image(corres_pos, camera_intrinsics_scene[camera_name2], camera_extrinsics_scene[camera_name2])
                left_corres_pos_overlay = create_point_overlay(*left_corres_pos_project, color=(0, 255, 0, 255))
                right_corres_pos_overlay = create_point_overlay(*right_corres_pos_project, color=(0, 255, 0, 255))
                left_corres_pos_offset_project = project_world_coord_to_image(corres_pos_offset, camera_intrinsics_scene[camera_name1], camera_extrinsics_scene[camera_name1])
                right_corres_pos_offset_project = project_world_coord_to_image(corres_pos_offset, camera_intrinsics_scene[camera_name2], camera_extrinsics_scene[camera_name2])
                left_corres_pos_offset_overlay = create_point_overlay(*left_corres_pos_offset_project, color=(255, 0, 0, 255))
                right_corres_pos_offset_overlay = create_point_overlay(*right_corres_pos_offset_project, color=(255, 0, 0, 255))
                left_image.paste(left_overlay, (0, 0), left_overlay)
                right_image.paste(right_overlay, (0, 0), right_overlay)
                left_image.paste(left_corres_point_overlay, (0, 0), left_corres_point_overlay)
                right_image.paste(right_corres_point_overlay, (0, 0), right_corres_point_overlay)
                left_image.paste(left_corres_pos_overlay, (0, 0), left_corres_pos_overlay)
                right_image.paste(right_corres_pos_overlay, (0, 0), right_corres_pos_overlay)
                left_image.paste(left_corres_pos_offset_overlay, (0, 0), left_corres_pos_offset_overlay)
                right_image.paste(right_corres_pos_offset_overlay, (0, 0), right_corres_pos_offset_overlay)
                triangulation_image = concatenate_images(left_image, right_image, text1=camera_name1, text2=camera_name2)
                triangulation_image.save(output_dir / f"triangulation_{keypoint_idx}_{i}.jpg")
            except:
                print(f"Failed to create triangulation visualization for correspondence {i}, keypoint {keypoint_idx}")
                continue
        best_i = warp_response[keypoint_idx]["best_candidate"]
        shutil.copyfile(output_dir / f"correspondence_{keypoint_idx}_{best_i}.jpg", output_dir / f"correspondence_{keypoint_idx}.jpg")
        shutil.copyfile(output_dir / f"triangulation_{keypoint_idx}_{best_i}.jpg", output_dir / f"triangulation_{keypoint_idx}.jpg")

