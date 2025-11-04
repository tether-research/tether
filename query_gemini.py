from PIL import Image
import string

from utils.annotation_utils import concatenate_images


def query_gemini_plan_actions(cfg, vlm, target_task, action_list, scene_dir, save_dir=None, retries=10):
    action_list = "\n".join([f"  {string.ascii_lowercase[i]}. " + action for i, action in enumerate(action_list)])
    for _ in range(retries):
        try:
            with open(cfg.prompt_path / "plan_actions.txt", "r") as f:
                determine_prompt = f.read()
                determine_prompt = determine_prompt.replace("<task>", target_task.rstrip("."))
                determine_prompt = determine_prompt.replace("<actions>", action_list)
            
            if save_dir is not None:
                save_path = save_dir / f"transcript_decision.md"
            else:
                save_path = None

            image_path = scene_dir / "formatted_image.jpg"
            img1, img2 = Image.open(scene_dir / "varied_camera_1.jpg"), Image.open(scene_dir / "varied_camera_2.jpg")
            img_formatted = concatenate_images(img1, img2, "left", "right")
            img_formatted.save(image_path)

            prompt = [determine_prompt, image_path]
            response = vlm.query(prompt, save_path=save_path, temperature=1.0, use_cache=False)
            response_json = vlm.parse_response_json(response)

            action_plan = [i["action"] for i in response_json["actions"]]
            return action_plan
        except Exception as e:
            print("Error in query_gemini_actions:", e)
            continue


def query_gemini_evaluate_success(cfg, vlm, trajectory_dir, action, keypoint_indices, output_dir):
    with open(cfg.prompt_path / "evaluate_success.txt", "r") as f:
        prompt = f.read()
        prompt = prompt.replace("<task>", action.rstrip("."))

    # Only use first and last keypoint
    first_keypoint = keypoint_indices[0]
    last_keypoint = keypoint_indices[-1]
    
    # Get raw frame paths for both cameras
    camera1_first = trajectory_dir / "recordings" / "frames" / "varied_camera_1" / f"{first_keypoint:05d}.jpg"
    camera2_first = trajectory_dir / "recordings" / "frames" / "varied_camera_2" / f"{first_keypoint:05d}.jpg"
    camera3_first = trajectory_dir / "recordings" / "frames" / "hand_camera" / f"{first_keypoint:05d}.jpg"
    camera1_last = trajectory_dir / "recordings" / "frames" / "varied_camera_1" / f"{last_keypoint:05d}.jpg"
    camera2_last = trajectory_dir / "recordings" / "frames" / "varied_camera_2" / f"{last_keypoint:05d}.jpg"
    camera3_last = trajectory_dir / "recordings" / "frames" / "hand_camera" / f"{last_keypoint:05d}.jpg"
    
    # Build prompt with raw individual images
    prompt = [prompt, camera1_first, camera2_first, camera3_first, camera1_last, camera2_last, camera3_last]
    
    print(f"Query gemini eval success (first/last) {trajectory_dir.name}")
    response = vlm.query(prompt, save_path=(output_dir / "transcript_eval_trajectory.md"), use_cache=False)
    response = vlm.parse_response_json(response)
    print(f"Response: {response}, {output_dir / 'transcript_eval_trajectory.md'}")
    return response

