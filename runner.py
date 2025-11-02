
from pathlib import Path
import shutil
import time
from tqdm import tqdm
import json
import random
import hydra
from omegaconf import DictConfig, OmegaConf
import threading
import wandb
import numpy as np
import concurrent.futures
import subprocess
from deepdiff import DeepDiff

from extract_keypoint_trajectory import extract_keypoint_trajectory 
from annotate_trajectory import annotate_demo_trajectory, annotate_rollout_trajectory
from query_gemini import query_gemini_evaluate_success, query_gemini_plan_actions
from run_correspondence import run_correspondence, create_correspondence_visualization, create_triangulation_visualization
from warp_trajectory import warp_trajectory
from ucb import UCB

from utils.vlm_utils import VlmWrapper
from utils.robot_utils import collect_scene_image, send_trajectory
from utils.timer_utils import Timer, timer
from utils.statistics_tracker import StatisticsTracker
from utils.misc_utils import is_valid_rollout, prepare_trajectory


class Runner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.vlm_smart, self.vlm_fast = self.init_vlm()
        self.action_library = {}
        self.ucb = UCB(save_path=self.cfg.exp_path / "ucb.json")
        self.ucb.load_state()

        self.cfg.demo_path.mkdir(exist_ok=True)
        self.cfg.scene_path.mkdir(exist_ok=True)
        self.cfg.rollout_path.mkdir(exist_ok=True)
        self.cfg.condense_path.mkdir(exist_ok=True)

        self.timer = Timer()
        self.stats = StatisticsTracker(self.cfg)
        self.action_id = {v: k for k, v in self.cfg.setting.demo_names.items()}

        self.rollout_processing_threads = {}
        self.refresh()

    def init_wandb(self):
        api = wandb.Api()
        prev_run = api.runs(
            f"upenn-pal/{self.cfg.run.wandb_project}",
            filters={"display_name": self.cfg.run.exp_name}
        )
        if len(prev_run) > 0:
            assert len(prev_run) == 1, f"Found {len(prev_run)} wandb runs with same name!"
            prev_run = prev_run[0]
            if not self.cfg.run.overwrite_cfg:
                prev_cfg = prev_run.config
                cur_cfg = OmegaConf.to_container(self.cfg, resolve=True, throw_on_missing=True)
                cur_cfg = {k: v if not isinstance(v, Path) else str(v) for k, v in cur_cfg.items()}
                diff = DeepDiff(prev_cfg, cur_cfg, ignore_order=True)
                if diff:
                    print("Config differences found:")
                    print(diff.to_json(indent=4))
                    raise ValueError("Previous run config does not match current config!")
            self.cfg.run.overwrite_cfg = False
            wandb_id, history = prev_run.id, prev_run.history()
            if not history.empty:
                self.stats.load(history)
        else:
            wandb_id = None
        wandb.init(
            project=self.cfg.run.wandb_project,
            name=self.cfg.run.exp_name,
            entity="upenn-pal",
            config=OmegaConf.to_container(self.cfg, resolve=True, throw_on_missing=True),
            resume="allow",
            id=wandb_id,
        )

    def init_vlm(self):
        vlm_smart = VlmWrapper(
            model=self.cfg.vlm.model_smart,
            api_key=self.cfg.vlm.api_key_smart,
            use_cache=self.cfg.cache_path,
            thinking_budget=self.cfg.vlm.thinking_budget_smart,
        )
        vlm_fast = VlmWrapper(
            model=self.cfg.vlm.model_fast,
            api_key=self.cfg.vlm.api_key_fast,
            use_cache=self.cfg.cache_path,
            thinking_budget=self.cfg.vlm.thinking_budget_fast,
        )
        return vlm_smart, vlm_fast
    
    def refresh(self):
        self.reload_action_library()
        for rollout, thread in list(self.rollout_processing_threads.items()):
            if not thread.is_alive():
                thread.join()
                del self.rollout_processing_threads[rollout]

    def reload_action_library(self):
        if not (self.cfg.exp_path / "action_library.json").exists():
            print("action_library.json does not exist, skipping reload")
            return

        while True:
            try:
                with open(self.cfg.exp_path / "action_library.json", "r") as f:
                    self.action_library = json.load(f)
                break
            except Exception:
                print("Failed to load action library, retrying...")
                time.sleep(1)
        for action, trajectories in self.action_library.items():
            for trajectory in trajectories:
                if (self.cfg.demo_path / trajectory).exists():
                    trajectory_path = self.cfg.demo_path / trajectory
                elif (self.cfg.rollout_path / trajectory).exists():
                    trajectory_path = self.cfg.rollout_path / trajectory
                else:
                    trajectory_path = Path(str(self.cfg.rollout_path).replace("single", "cycle")) / trajectory
                with open(trajectory_path / "metadata.json", "r") as f:
                    metadata = json.load(f)
                metadata["action"] = action
                with open(trajectory_path / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=4)
                self.ucb.add_arm(f"{self.action_id[action]}/{trajectory}")
            self.stats.set(f"action_library_size/{self.action_id[action]}", len(trajectories))
        
    def invalidate_corrupted_rollouts(self):
        for rollout_dir in self.cfg.rollout_path.iterdir():
            if rollout_dir.name in self.rollout_processing_threads:
                continue
            if not is_valid_rollout(rollout_dir) and not rollout_dir.name.endswith(".invalid"):
                rollout_dir.rename(rollout_dir.with_suffix(".invalid"))

    def rebuild_action_library(self, wait_for_rollouts=True):
        if wait_for_rollouts:
            for rollout, thread in list(self.rollout_processing_threads.items()):
                thread.join()
                del self.rollout_processing_threads[rollout]
            self.invalidate_corrupted_rollouts()

        self.reload_action_library()
        for rollout in self.cfg.rollout_path.iterdir():
            if rollout.name.endswith(".invalid") or rollout.name in self.rollout_processing_threads:
                continue
            with open(rollout / "metadata.json", "r") as f:
                metadata = json.load(f)
                success, action = metadata["success"], metadata["action"]
            if success:
                if action not in self.action_library:
                    print(f"WARNING: rollout {rollout.name} has action {action} not in action library")
                    continue
                if rollout.name not in self.action_library[action]:
                    self.action_library[action].append(rollout.name)
        if (self.cfg.exp_path / "action_library.json").exists():
            wandb.save(str(self.cfg.exp_path / "action_library.json"), base_path=str(self.cfg.exp_path))
            (self.cfg.exp_path / "action_library.json").rename(self.cfg.exp_path / f"action_library.json.old")
        with open(self.cfg.exp_path / "action_library.json", "w") as f:
            json.dump(self.action_library, f, indent=4)
        self.reload_action_library()

    def prepare_bootstrap(self):
        self.object_names = self.cfg.setting.object_names
        self.action_library = {}
        for multi_demo_dir, action in self.cfg.setting.demo_names.items():
            self.action_library[action] = []
            for demo_dir in tqdm(list((self.cfg.global_demo_path / multi_demo_dir).iterdir())):
                self.action_library[action].append(demo_dir.name)
                if (self.cfg.demo_path / demo_dir.name).exists():
                    continue
                shutil.copytree(demo_dir, self.cfg.demo_path / demo_dir.name)
                demo_dir = self.cfg.demo_path / demo_dir.name
                prepare_trajectory(self.cfg, demo_dir, direction=1, output_dir=demo_dir)
                extract_keypoint_trajectory(self.cfg, demo_dir, output_dir=demo_dir)
                annotate_demo_trajectory(self.cfg, demo_dir, output_dir=demo_dir)
                metadata = {
                    "action": action,
                    "articulated": False,
                }
                with open(demo_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=4)
        if (self.cfg.exp_path / "action_library.json").exists():
            (self.cfg.exp_path / "action_library.json").rename(self.cfg.exp_path / f"action_library.json.old")
        with open(self.cfg.exp_path / "action_library.json", "w") as f:
            json.dump(self.action_library, f, indent=4)
    
    def warp_trajectory(self, demo_dir, scene_dir):
        pipeline_dir = demo_dir / "pipeline"
        if pipeline_dir.exists():
            shutil.rmtree(pipeline_dir)
        pipeline_dir.mkdir()

        prepare_trajectory(self.cfg, demo_dir, direction=1, output_dir=demo_dir)
        keypoint_indices, open_indices, close_indices = extract_keypoint_trajectory(self.cfg, demo_dir, output_dir=demo_dir)
        gripper_indices = np.sort(np.concatenate((open_indices, close_indices)))
        warp = run_correspondence(self.cfg, gripper_indices, demo_dir, scene_dir, output_dir=pipeline_dir)
        if warp is None:
            return
        try:
            warped_trajectory, warping_infos = warp_trajectory(self.cfg, demo_dir, output_dir=pipeline_dir)
        except Exception as e:
            print(f"Error while warping trajectory: {e}")
            return
        return warped_trajectory, warping_infos, warp
    
    @timer
    def generate_action_plan(self, scene_dir, target_task):
        while True:
            try:
                action_plan = query_gemini_plan_actions(self.cfg, self.vlm_fast, target_task, list(self.action_library.keys()), self.object_names, scene_dir, save_dir=self.cfg.exp_path)
                return action_plan
            except Exception as e:
                print(f"Error while generating action plan: {e}")
                print("Retrying...")
                time.sleep(1)
    
    def process_rollout(self, demo_dir, scene_dir, remote_rollout_dir, action, evaluate=True):
        shutil.move(demo_dir / "pipeline", self.cfg.rollout_path / f"pipeline_{demo_dir.name}")  # Avoid race condition with next iteration
        subprocess.run(["rsync", "-avz", "--exclude='*.svo2'", "--exclude='*.jpg'", f"eth.franka.franka-laptop:{remote_rollout_dir}", f"{self.cfg.rollout_path}"], check=True, stdout=subprocess.DEVNULL)
        rollout_dir = self.cfg.rollout_path / remote_rollout_dir.name

        create_correspondence_visualization(self.cfg, demo_dir, scene_dir, output_dir=self.cfg.rollout_path / f"pipeline_{demo_dir.name}")
        create_triangulation_visualization(self.cfg, demo_dir, scene_dir, output_dir=self.cfg.rollout_path / f"pipeline_{demo_dir.name}")

        shutil.move(self.cfg.rollout_path / f"pipeline_{demo_dir.name}", rollout_dir / f"pipeline_{demo_dir.name}")
        for camera in list(self.cfg.setting.cameras.keys()) + list(self.cfg.setting.hand_cameras.keys()):
            camera_dir = rollout_dir / "recordings" / "frames" / camera
            video_path = rollout_dir / "recordings" / f"{camera}.mp4"
            command = f"ffmpeg -ss 0 -i {video_path} -qscale:v 2 -vf fps=15 -start_number 0 {camera_dir}/%05d.jpg"
            subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if (self.cfg.exp_path / "transcript_decision.md").exists():
            (self.cfg.exp_path / "transcript_decision.md").rename(rollout_dir / f"transcript_decision.md")
        prepare_trajectory(self.cfg, rollout_dir, direction=1, output_dir=rollout_dir)
        keypoint_indices, _, _ = extract_keypoint_trajectory(self.cfg, rollout_dir, output_dir=rollout_dir)
        annotate_rollout_trajectory(self.cfg, rollout_dir, demo_dir, output_dir=rollout_dir)

        if evaluate:
            eval_dir = rollout_dir / "evaluation"
            eval_dir.mkdir(exist_ok=True)
            demo_name = demo_dir.name
            try:
                evaluation_response = query_gemini_evaluate_success(self.cfg, self.vlm_smart, rollout_dir, action, self.object_names, keypoint_indices, output_dir=eval_dir)
                success = evaluation_response["completed"]
            except Exception as e:
                print(f"Error while evaluating rollout: {e}")
                success = False
            with open(demo_dir / "metadata.json", "r") as f:
                demo_metadata = json.load(f)
            metadata = {
                "action": action,
                "success": success,
                "demo": demo_name,
                "articulated": demo_metadata["articulated"],
            }
            with open(rollout_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=4)
            print(f"!!!!!!!!!!!!!! Finished processing {rollout_dir.name} ({action}): success {success} !!!!!!!!!!!!!!")
            if success:
                self.stats.add(f"action_success/{self.action_id[action]}")
                self.stats.add(f"demo_success/{self.action_id[action]}/{demo_name}")
            self.ucb.update_arm(f"{self.action_id[action]}/{demo_name}", success)

    def process_rollout_async(self, demo_dir, scene_dir, remote_rollout_dir, action, evaluate=True):
        thread = threading.Thread(target=self.process_rollout, args=(demo_dir, scene_dir, remote_rollout_dir, action, evaluate))
        thread.start()
        self.rollout_processing_threads[remote_rollout_dir.name] = thread

    def run_single(self, action):
        self.prepare_bootstrap()

        while True:
            self.timer.start("iteration")
            scene_name = collect_scene_image(self.cfg)
            if scene_name is None:
                print("Failed to prepare scene! Are the cameras connected?")
                return
            scene_dir = self.cfg.scene_path / scene_name

            demo_arms = self.action_library[action]
            random.shuffle(demo_arms)

            print(f"************** {action} **************")
            print(f"Scene: {scene_name}")

            candidate_warpings = []
            for i in range(len(demo_arms)):
                demo_name = demo_arms[i]
                print(f"Running pipeline with trajectory {demo_name}")
                demo_dir = self.cfg.demo_path / demo_name if (self.cfg.demo_path / demo_name).exists() else self.cfg.rollout_path / demo_name
                warp_result = self.warp_trajectory(demo_dir, scene_dir)
                if warp_result is None:
                    # For debugging
                    # create_correspondence_visualization(self.cfg, demo_dir, scene_dir, output_dir=demo_dir / f"pipeline")
                    # create_triangulation_visualization(self.cfg, demo_dir, scene_dir, output_dir=demo_dir / f"pipeline")
                    # import ipdb; ipdb.set_trace()

                    shutil.rmtree(demo_dir / "pipeline")
                    continue
                warped_trajectory, warping_infos, warp = warp_result
                candidate_warpings.append((demo_name, warped_trajectory, warping_infos, warp))
                if self.cfg.num_warp_attempts == 0 or i >= self.cfg.num_warp_attempts:
                    break
            
            if len(candidate_warpings) == 0:
                continue

            best_i, best_warp_distance = -1, float("inf")
            for i, (demo_name, warped_trajectory, warping_infos, warp) in enumerate(candidate_warpings):
                warp_distance = sum([np.linalg.norm(warp[keypoint_idx]["position_delta"]) for keypoint_idx in warp])
                if warp_distance < best_warp_distance:
                    best_i, best_warp_distance = i, warp_distance
            demo_name, warped_trajectory, warping_infos, warp = candidate_warpings[best_i]
            demo_dir = self.cfg.demo_path / demo_name if (self.cfg.demo_path / demo_name).exists() else self.cfg.rollout_path / demo_name

            print(f"Selected demo {demo_name}, running trajectory...")
            prepare_trajectory(self.cfg, demo_dir, direction=-1, output_dir=demo_dir)
            remote_rollout_dir = send_trajectory(self.cfg, demo_dir, action, save=False)
            if remote_rollout_dir is not None:
                self.process_rollout_async(demo_dir, scene_dir, remote_rollout_dir, action, evaluate=False)
            else:
                print(f"Failed to send trajectory for {demo_name}!")
            
            self.timer.end("iteration")
            input("Finished rollout, press any key to continue...")
        
    def run_cycle(self):
        self.prepare_bootstrap()
        self.init_wandb()

        it = 0
        while True:
            self.timer.start("iteration")
            if self.cfg.library_rebuild_interval > 0 and it % self.cfg.library_rebuild_interval == 0:
                print("Rebuilding action library...")
                self.rebuild_action_library()

            print("==================================================")

            self.refresh()

            scene_name = collect_scene_image(self.cfg)
            if scene_name is None:
                print("Failed to prepare scene! Are the cameras connected?")
                return
            scene_dir = self.cfg.scene_path / scene_name

            action_list = list(self.action_library.keys())
            selection_weights = np.array([self.stats.get(f"action_success/{self.action_id[action]}") for action in action_list])
            selection_weights = np.exp(-selection_weights / 10)
            selection_probs = np.array(selection_weights) / np.sum(selection_weights)
            target_tasks = np.random.choice(action_list, size=min(len(action_list), self.cfg.num_planners), replace=False, p=selection_probs)
            for i, action in enumerate(action_list):
                self.stats.set(f"target_prob/{self.action_id[action]}", selection_probs[i])

            print(f"Querying action plan for {len(target_tasks)} tasks:")
            print("\n".join([f"  - {action} ({self.action_id[action]})" for action in target_tasks]))
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(target_tasks)) as executor:
                futures = {executor.submit(self.generate_action_plan, scene_dir, task): task for task in target_tasks}
                for task in target_tasks:
                    self.stats.add(f"target_queried/{self.action_id[task]}")
                for future in concurrent.futures.as_completed(futures):
                    target_task = futures[future]
                    self.stats.add(f"target_selected/{self.action_id[target_task]}")
                    action_plan = future.result()

                    if len(action_plan) == 0:
                        print(f"Action plan for {target_task} is empty, skipping...")
                        continue

                    action = action_plan[0]
                    if action_plan[0] not in self.action_library:
                        print(f"Action ({action}) not in action library, skipping...")
                        continue

                    valid_demo_arms = [f"{self.action_id[action]}/{name}" for name in self.action_library[action]]
                    _, demo_arms, demo_scores = self.ucb.select_arm(valid_arms=valid_demo_arms)

                    print(f"************** {action} **************")
                    print(f"Target task: {target_task}")
                    print(f"Scene: {scene_name}")

                    candidate_warpings = []
                    for i in range(len(demo_arms)):
                        demo_arm = demo_arms[i]
                        demo_name = demo_arm.split("/")[1]

                        self.stats.add(f"action_attempted/{self.action_id[action]}")
                        self.stats.add(f"demo_attempted/{self.action_id[action]}/{demo_name}")

                        print(f"Running pipeline with trajectory {demo_name}")
                        demo_dir = self.cfg.demo_path / demo_name if (self.cfg.demo_path / demo_name).exists() else self.cfg.rollout_path / demo_name
                        warp_result = self.warp_trajectory(demo_dir, scene_dir)
                        if warp_result is None:
                            self.ucb.update_arm(f"{self.action_id[action]}/{demo_name}", False)
                            self.stats.add(f"demo_corres_failed/{self.action_id[action]}/{demo_name}")
                            shutil.rmtree(demo_dir / "pipeline")
                        else:
                            warped_trajectory, warping_infos, warp = warp_result
                            candidate_warpings.append((demo_name, warped_trajectory, warping_infos, warp))
                        if self.cfg.num_warp_attempts == 0 or i + 1 >= self.cfg.num_warp_attempts:
                            break
                    
                    if len(candidate_warpings) == 0:
                        continue

                    best_i, best_warp_distance = -1, float("inf")
                    for i, (demo_name, warped_trajectory, warping_infos, warp) in enumerate(candidate_warpings):
                        warp_distance = sum([np.linalg.norm(warp[keypoint_idx]["position_delta"]) for keypoint_idx in warp])
                        if warp_distance < best_warp_distance:
                            best_i, best_warp_distance = i, warp_distance
                    demo_name, warped_trajectory, warping_infos, warp = candidate_warpings[best_i]
                    demo_dir = self.cfg.demo_path / demo_name if (self.cfg.demo_path / demo_name).exists() else self.cfg.rollout_path / demo_name

                    print(f"Selected demo {demo_name}, running trajectory...")
                    prepare_trajectory(self.cfg, demo_dir, direction=-1, output_dir=demo_dir)
                    remote_rollout_dir = send_trajectory(self.cfg, demo_dir, action, save=False)
                    if remote_rollout_dir is not None:
                        self.process_rollout_async(demo_dir, scene_dir, remote_rollout_dir, action)
                        self.stats.add(f"action_executed/{self.action_id[action]}")
                        self.stats.add(f"demo_executed/{self.action_id[action]}/{demo_name}")
                    else:
                        print(f"Failed to send trajectory for {demo_name}!")
                    break

            time = self.timer.end("iteration")
            self.stats.set("timing/iteration", time)
            self.stats.add("timing/total", time)
            self.stats.log()
            it += 1
    

@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_pipeline(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    cfg.exp_path = Path(cfg.exp_path).resolve()
    cfg.global_demo_path = Path(cfg.global_demo_path).resolve()
    cfg.demo_path = Path(cfg.demo_path).resolve()
    cfg.scene_path = Path(cfg.scene_path + f"_{cfg.mode}").resolve()
    cfg.rollout_path = Path(cfg.rollout_path + f"_{cfg.mode}").resolve()
    cfg.condense_path = Path(cfg.condense_path).resolve()
    cfg.cache_path = Path(cfg.cache_path).resolve()
    cfg.prompt_path = Path(cfg.prompt_path).resolve()

    runner = Runner(cfg)
    if cfg.mode == "cycle":
        runner.run_cycle()
    elif cfg.mode == "single":
        runner.run_single(cfg.action)
    else:
        raise ValueError(f"Unknown mode {cfg.mode}")

if __name__ == '__main__':
    run_pipeline()
