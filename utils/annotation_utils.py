import numpy as np
import cv2
import scipy.interpolate
import matplotlib.pyplot as plt
from PIL import Image

from utils.geometry_utils import project_camera_to_image, euler_to_rmat
from utils.drawing_utils import draw_circle_text, draw_axis, draw_gripper, draw_circle_text_semitransparent, draw_line_semitransparent
from utils.misc_utils import modify_trajectory_position


def create_label_overlay(texts):
    texts = [(texts, (255, 255, 255, 255), (0, 0, 0, 255))] if isinstance(texts, str) else texts
    label_overlay = np.zeros((720, 1280, 4), dtype=np.uint8)
    y = 0
    for text, text_color, bg_color in texts:
        text_width, text_height = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        cv2.rectangle(label_overlay, (0, y), (text_width + 10, y + text_height + 10), bg_color, -1)
        cv2.putText(label_overlay, text, (5, y + text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 3)
        y += text_height + 10
    return Image.fromarray(label_overlay)


def create_trajectory_overlay(camera_frame_trajectory, camera_intrinsics, gripper_len, densify_factor=64):
    def create_spline(trajectory):
        x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
        t = np.linspace(0, 1, len(trajectory))
        tck, _ = scipy.interpolate.splprep([x, y, z], u=t, s=0)
        return tck
    
    camera_frame_trajectory = modify_trajectory_position(camera_frame_trajectory, direction=1, magnitude=gripper_len)

    height, width = 720, 1280
    trajectory_overlay = np.zeros((height, width, 4), dtype=np.uint8)
    total_points = len(camera_frame_trajectory) * densify_factor
    cmap = plt.cm.rainbow(np.linspace(0, 1, total_points))
    camera_frame_spline_trajectory = create_spline(camera_frame_trajectory)
    t_values = np.linspace(0, 1, total_points)
    positions = np.array(scipy.interpolate.splev(t_values, camera_frame_spline_trajectory))
    pixels = [project_camera_to_image(positions[:, i], camera_intrinsics)
              for i in range(total_points)]
    for pixel, color_val in zip(pixels, cmap):
        x, y = int(pixel[0]), int(pixel[1])
        if 0 <= x < width and 0 <= y < height:
            color = tuple(map(int, color_val * 255))
            cv2.circle(trajectory_overlay, (x, y), 2, color, -1)
    
    def trajectory_animator():
        cur_trajectory_overlay = np.zeros((height, width, 4), dtype=np.uint8)
        for i, (pixel, color_val) in enumerate(zip(pixels, cmap)):
            x, y = int(pixel[0]), int(pixel[1])
            if 0 <= x < width and 0 <= y < height:
                color = tuple(map(int, color_val * 255))
                cv2.circle(cur_trajectory_overlay, (x, y), 2, color, -1)
            if i % densify_factor == densify_factor - 1:
                yield Image.fromarray(cur_trajectory_overlay)
    
    return Image.fromarray(trajectory_overlay), trajectory_animator()


def create_keypoint_overlays(trajectory, keypoint_indices, camera_intrinsics, gripper_len, end_timestep=None):
    trajectory = modify_trajectory_position(trajectory, direction=1, magnitude=gripper_len)
    keypoint_overlay = Image.fromarray(np.zeros((720, 1280, 4), dtype=np.uint8))
    keypoint_overlays = []
    cmap = plt.cm.rainbow(np.linspace(0, 1, len(trajectory)))
    for i, idx in enumerate(keypoint_indices):
        if end_timestep is not None and idx > end_timestep:
            break
        color = tuple(map(int, cmap[idx] * 255))
        cur_keypoint_overlay = np.zeros((720, 1280, 4), dtype=np.uint8)
        pixel = project_camera_to_image(trajectory[idx, :3], camera_intrinsics)
        x, y = int(pixel[0]), int(pixel[1])
        if 0 <= x < 1280 and 0 <= y < 720:
            draw_circle_text(cur_keypoint_overlay, x, y, f"{i:02d}", color=color)
        cur_keypoint_overlay = Image.fromarray(cur_keypoint_overlay)
        keypoint_overlays.append(cur_keypoint_overlay)
        keypoint_overlay.paste(cur_keypoint_overlay, (0, 0), cur_keypoint_overlay)
    
    def keypoint_animator():
        cur_keypoint_idx = 0
        cur_keypoint_overlay = Image.fromarray(np.zeros((720, 1280, 4), dtype=np.uint8))
        for i in range(len(trajectory)):
            if cur_keypoint_idx < len(keypoint_indices) and i == keypoint_indices[cur_keypoint_idx]:
                cur_keypoint_idx += 1
            if cur_keypoint_idx > 0:
                cur_keypoint_overlay.paste(keypoint_overlays[cur_keypoint_idx-1], (0, 0), keypoint_overlays[cur_keypoint_idx-1])
            yield cur_keypoint_overlay

    return keypoint_overlay, keypoint_overlays, keypoint_animator()


def create_axis_overlays(camera_frame_trajectory, camera_intrinsics):
    axis_overlays = []
    for i in range(len(camera_frame_trajectory)):
        axis_overlay = np.zeros((720, 1280, 4), dtype=np.uint8)
        axis_overlay = draw_axis(axis_overlay, camera_frame_trajectory[i][:3], camera_frame_trajectory[i][3:6], camera_intrinsics)
        axis_overlay = Image.fromarray(axis_overlay)
        axis_overlays.append(axis_overlay)
    return axis_overlays


def create_gripper_overlays(camera_frame_trajectory, camera_intrinsics, gripper_len, color=(255, 255, 0, 255)):
    gripper_overlays = []
    for i in range(len(camera_frame_trajectory)):
        gripper_overlay = np.zeros((720, 1280, 4), dtype=np.uint8)
        gripper_overlay = draw_gripper(gripper_overlay, camera_frame_trajectory[i][:3], camera_frame_trajectory[i][3:6], camera_frame_trajectory[i][6], camera_intrinsics, gripper_len, color=color)
        gripper_overlay = Image.fromarray(gripper_overlay)
        gripper_overlays.append(gripper_overlay)
    return gripper_overlays


def create_position_marker_overlay(camera_frame_trajectory, camera_intrinsics, text, color=(255, 255, 255, 255)):
    marker_overlays = []
    gripper_end_pt = np.array([0.0, 0.0, 0.15])
    for i in range(len(camera_frame_trajectory)):
        position_marker_overlay = np.zeros((720, 1280, 4), dtype=np.uint8)
        pos =  camera_frame_trajectory[i][:3]
        rot = camera_frame_trajectory[i][3:6]
        gripper_end_pt_3d = pos + euler_to_rmat(rot) @ gripper_end_pt
        gripper_end_pt_pixel = project_camera_to_image(gripper_end_pt_3d, camera_intrinsics)
        marker_overlay = draw_circle_text_semitransparent(position_marker_overlay, int(gripper_end_pt_pixel[0]), int(gripper_end_pt_pixel[1]), text, 6, color=color, radius=22)
        marker_overlay = Image.fromarray(marker_overlay)
        marker_overlays.append(marker_overlay)
    return marker_overlays


def create_epipolar_line_overlay(start_pixel, direction, color=(255, 255, 255, 255)):
    assert direction is not None, f"Direction vector cannot be None"
    def get_intersection_with_image_boundary_outward(start_pixel, direction, img_height, img_width):
        x0, y0 = start_pixel
        dx, dy = direction
        if x0 < 0 or x0 >= img_width or y0 < 0 or y0 >= img_height:
            return (int(round(x0)), int(round(y0)))
        if dx == 0 and dy == 0:
            return None

        candidates = []
        def check_and_add(t_value, boundary_x, boundary_y):
            if t_value > 0:
                ix = boundary_x if boundary_x is not None else x0 + t_value*dx
                iy = boundary_y if boundary_y is not None else y0 + t_value*dy
                if 0 <= ix <= img_width - 1 and 0 <= iy <= img_height - 1:
                    candidates.append((t_value, (int(round(ix)), int(round(iy)))))

        if dx != 0:
            t = (0 - x0) / dx
            check_and_add(t, boundary_x=0, boundary_y=None)
        if dx != 0:
            t = ((img_width - 1) - x0) / dx
            check_and_add(t, boundary_x=img_width - 1, boundary_y=None)
        if dy != 0:
            t = (0 - y0) / dy
            check_and_add(t, boundary_x=None, boundary_y=0)
        if dy != 0:
            t = ((img_height - 1) - y0) / dy
            check_and_add(t, boundary_x=None, boundary_y=img_height - 1)
        if candidates:
            candidates.sort(key=lambda c: c[0])
            return candidates[0][1]
        candidates_neg = []
        def check_and_add_neg(t_value, boundary_x, boundary_y):
            if t_value < 0:
                ix = boundary_x if boundary_x is not None else x0 + t_value*dx
                iy = boundary_y if boundary_y is not None else y0 + t_value*dy
                if 0 <= ix <= img_width - 1 and 0 <= iy <= img_height - 1:
                    candidates_neg.append((t_value, (int(round(ix)), int(round(iy)))))

        if dx != 0:
            t = (0 - x0) / dx
            check_and_add_neg(t, boundary_x=0, boundary_y=None)
            t = ((img_width - 1) - x0) / dx
            check_and_add_neg(t, boundary_x=img_width - 1, boundary_y=None)
        if dy != 0:
            t = (0 - y0) / dy
            check_and_add_neg(t, boundary_x=None, boundary_y=0)
            t = ((img_height - 1) - y0) / dy
            check_and_add_neg(t, boundary_x=None, boundary_y=img_height - 1)

        if not candidates_neg:
            return None

        candidates_neg.sort(key=lambda c: abs(c[0]))
        return candidates_neg[0][1]

    def get_intersection_with_image_boundary_inward(start_pixel, direction, img_height, img_width):
        assert direction is not None, f"Direction vector cannot be None. Direction: {direction}"
        x0, y0 = start_pixel
        dx, dy = direction
        if 0 <= x0 < img_width and 0 <= y0 < img_height:
            return (int(round(x0)), int(round(y0)))
        if dx == 0 and dy == 0:
            return None
        
        candidates = []
        def check_and_add(t_value, boundary_x, boundary_y):
            if t_value >= 0:
                ix = boundary_x if boundary_x is not None else x0 + t_value*dx
                iy = boundary_y if boundary_y is not None else y0 + t_value*dy
                if 0 <= ix <= img_width-1 and 0 <= iy <= img_height-1:
                    candidates.append((t_value, (int(round(ix)), int(round(iy)))))
        
        if dx != 0:
            t = (0 - x0) / dx
            check_and_add(t, boundary_x=0, boundary_y=None)
        if dx != 0:
            t = ((img_width - 1) - x0) / dx
            check_and_add(t, boundary_x=img_width - 1, boundary_y=None)
        if dy != 0:
            t = (0 - y0) / dy
            check_and_add(t, boundary_x=None, boundary_y=0)
        if dy != 0:
            t = ((img_height - 1) - y0) / dy
            check_and_add(t, boundary_x=None, boundary_y=img_height - 1)
        if candidates:
            candidates.sort(key=lambda c: abs(c[0]))  # might pick the intersection closest to t=0
            return candidates[0][1]
        
        candidates_neg = []
        def check_and_add_neg(t_value, boundary_x, boundary_y):
            if t_value < 0:  # i.e. backward
                ix = boundary_x if boundary_x is not None else x0 + t_value*dx
                iy = boundary_y if boundary_y is not None else y0 + t_value*dy
                if 0 <= ix < img_width and 0 <= iy < img_height:
                    candidates_neg.append((t_value, (int(round(ix)), int(round(iy)))))
        
        if dx != 0:
            t = (0 - x0) / dx
            check_and_add_neg(t, boundary_x=0, boundary_y=None)
            t = ((img_width - 1) - x0) / dx
            check_and_add_neg(t, boundary_x=img_width - 1, boundary_y=None)
        if dy != 0:
            t = (0 - y0) / dy
            check_and_add_neg(t, boundary_x=None, boundary_y=0)
            t = ((img_height - 1) - y0) / dy
            check_and_add_neg(t, boundary_x=None, boundary_y=img_height - 1)

        if not candidates_neg:
            return None
        candidates_neg.sort(key=lambda c: c[0])
        return candidates_neg[0][1]

    direction = direction / np.linalg.norm(direction)
    epipolar_line_start = get_intersection_with_image_boundary_inward(start_pixel, direction, 720, 1280) 
    epipolar_line_start = np.array(epipolar_line_start)

    marker_overlay = np.zeros((720, 1280, 4), dtype=np.uint8)
    epipolar_line_end = get_intersection_with_image_boundary_outward(epipolar_line_start, direction, 720, 1280)
    epipolar_line_end = np.array(epipolar_line_end)
    draw_line_semitransparent(marker_overlay, tuple(map(int, epipolar_line_start)), tuple(map(int, epipolar_line_end)), color, thickness=8, alpha=0.7)
    return Image.fromarray(marker_overlay)


def create_point_overlay(x, y, size=4, color=(255, 0, 0, 255)):
    x, y = int(x), int(y)
    point_overlay = np.zeros((720, 1280, 4), dtype=np.uint8)
    cv2.circle(point_overlay, (x, y), size, color, -1)
    return Image.fromarray(point_overlay)


def create_numbered_point_overlay(point_overlay, x, y, number, size=10, color=(255, 255, 255, 128), text_color=(0, 0, 0, 255)):
    height, width = 720, 1280
    
    x, y = int(x), int(y)
    if 0 <= x < width and 0 <= y < height:
        cv2.circle(point_overlay, (x, y), size, color, -1)
        text_size = cv2.getTextSize(str(number), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = x - text_size[0]//2
        text_y = y + text_size[1]//2
        cv2.putText(point_overlay, str(number), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color[:3], 2)


def create_x_overlay(x, y, size=4, color=(255, 0, 0, 255)):
    x, y = int(x), int(y)
    overlay = np.zeros((720, 1280, 4), dtype=np.uint8)
    line1_start = (x - size, y - size)
    line1_end = (x + size, y + size)
    line2_start = (x - size, y + size)
    line2_end = (x + size, y - size)
    cv2.line(overlay, line1_start, line1_end, color, thickness=1)
    cv2.line(overlay, line2_start, line2_end, color, thickness=1)
    return Image.fromarray(overlay)
 

def create_bbox_overlay(x1, y1, x2, y2, color=(255, 0, 0, 255)):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    bbox_overlay = np.zeros((720, 1280, 4), dtype=np.uint8)
    cv2.rectangle(bbox_overlay, (x1, y1), (x2, y2), color, 2)
    return Image.fromarray(bbox_overlay)


def concatenate_images(img1, img2, text1=None, text2=None, orientation="horizontal"):
    img1, img2 = np.array(img1), np.array(img2)
    if orientation == "horizontal":
        height = min(img1.shape[0], img2.shape[0])
        img1 = cv2.resize(img1, (int(img1.shape[1] * height / img1.shape[0]), height))
        img2 = cv2.resize(img2, (int(img2.shape[1] * height / img2.shape[0]), height))
        combined_image = np.hstack((img1, img2))
        text_positions = [(0, 0), (img1.shape[1], 0)]
    elif orientation == "vertical":
        width = min(img1.shape[1], img2.shape[1])
        img1 = cv2.resize(img1, (width, int(img1.shape[0] * width / img1.shape[1])))
        img2 = cv2.resize(img2, (width, int(img2.shape[0] * width / img2.shape[1])))
        combined_image = np.vstack((img1, img2))
        text_positions = [(0, 0), (0, img1.shape[0])]

    text_color = (255, 255, 255)
    bg_color = (0, 0, 0)
    for (text, pos) in zip([text1, text2], text_positions):
        if text:
            x, y = pos
            text_width, text_height = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            cv2.rectangle(combined_image, (x, y), (x + text_width + 10, y + text_height + 10), bg_color, -1)
            cv2.putText(combined_image, text, (x + 5, y + text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 3)
    return Image.fromarray(combined_image)


def crop_image(img, output_path=None, crop_ratio=(0, 0, 0, 0)):
    w, h = img.size
    left = crop_ratio[1] * w
    top = crop_ratio[0] * h
    right = w - crop_ratio[3] * w
    bottom = h - crop_ratio[2] * h
    img = img.crop((left, top, right, bottom))
    if output_path:
        img.save(output_path)
    return img


def flip_image(img, output_path=None, flip_direction="vertical"):
    if flip_direction == "vertical":
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    elif flip_direction == "horizontal":
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        raise ValueError(f"Invalid flipping direction: {flip_direction}")
    if output_path:
        img.save(output_path)
    return img

