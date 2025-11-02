import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils.geometry_utils import euler_to_rmat, project_camera_to_image


def draw_axis(img, pos, rot, intrinsics):
    axis_length = 0.1  # meters
    tick_size = 8      # pixels
    tick_positions = [0.025, 0.05, 0.075]
    axes = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    colors = [
        (255, 0, 0, 255),
        (0, 255, 0, 255),
        (0, 0, 255, 255)
    ]

    origin_pixel = project_camera_to_image(pos, intrinsics)
    origin_pixel = tuple(map(int, origin_pixel))
    
    for axis, color in zip(axes, colors):
        endpoint = pos + euler_to_rmat(rot) @ axis * axis_length
        endpoint_pixel = project_camera_to_image(endpoint, intrinsics)
        endpoint_pixel = tuple(map(int, endpoint_pixel))
        cv2.arrowedLine(img, origin_pixel, endpoint_pixel, color, 2, tipLength=0.2)

        for tick in tick_positions:
            tick_point = pos + euler_to_rmat(rot) @ axis * tick
            tick_pixel = project_camera_to_image(tick_point, intrinsics)
            prep_dx, prep_dy = endpoint_pixel[0] - origin_pixel[0], endpoint_pixel[1] - origin_pixel[1]
            perp_unit = np.array([-prep_dy, prep_dx]) / np.linalg.norm([prep_dx, prep_dy])

            tick_endpoint1 = tick_pixel + perp_unit * tick_size
            tick_endpoint2 = tick_pixel - perp_unit * tick_size
            cv2.line(img, tuple(map(int, tick_endpoint1)), tuple(map(int, tick_endpoint2)), color, 2)

    return img


def draw_gripper(img, pos, rot, gripper, intrinsics, gripper_len, color=(255, 255, 0, 255)):
    gripper_open_width, gripper_close_width = 0.07, 0.02  # meters
    gripper_width = gripper * gripper_close_width + (1 - gripper) * gripper_open_width
    base, hinge, fingertip = 0.0, 0.05, gripper_len
    gripper_lines = np.array([
        [[0.0, gripper_width, hinge], [0.0, gripper_width, fingertip]],
        [[0.0, -gripper_width, hinge], [0.0, -gripper_width, fingertip]],
        [[0.0, gripper_width, hinge], [0.0, -gripper_width, hinge]],
        [[0.0, 0.0, base], [0.0, 0.0, hinge]],
    ])

    for (gripper_point_1, gripper_point_2) in gripper_lines:
        gripper_point_1 = pos + euler_to_rmat(rot) @ gripper_point_1
        gripper_pixel_1 = project_camera_to_image(gripper_point_1, intrinsics)
        gripper_point_2 = pos + euler_to_rmat(rot) @ gripper_point_2
        gripper_pixel_2 = project_camera_to_image(gripper_point_2, intrinsics)
        cv2.line(img, tuple(map(int, gripper_pixel_1)), tuple(map(int, gripper_pixel_2)), color, 4)

    return img


def draw_circle_text(img, x, y, text, color=(255, 255, 255, 255), text_color=None):
    if text_color is None:
        text_color = (0, 0, 0, 255) if np.mean(color[:3]) > 128 else (255, 255, 255, 255)
    cv2.circle(img, (x, y), 12, color, -1)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    text_x, text_y = x - text_size[0] // 2, y + text_size[1] // 2
    cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    return img


def draw_circle_text_semitransparent(img, x, y, color=(255, 255, 255, 255), text_color=(0, 0, 0, 255), radius=30):
    overlay = img.copy()
    cv2.circle(overlay, (x, y), radius, (255, 255, 255, 255), -1)
    alpha = 0.3
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.circle(img, (x, y), radius, color, 2)
    text = "0"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    text_x, text_y = x - text_size[0] // 2, y + text_size[1] // 2
    cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    return img


def draw_line_semitransparent(img, pt1, pt2, color, thickness=8, alpha=0.5):
    overlay = img.copy()
    cv2.line(overlay, pt1, pt2, color, thickness)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)


def draw_number(img, x, y, number):
    cv2.putText(img, str(number), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return img


def draw_point(img, x, y, size=4, color=(0, 0, 255)):
    cv2.circle(img, (x, y), size, color, -1)
    return img


def show_2_lines_and_intersection(line1_start, line1_dir, line2_start, line2_dir, mean_pt):
    line1_dir = line1_dir / np.linalg.norm(line1_dir) * 2
    line2_dir = line2_dir / np.linalg.norm(line2_dir) * 2
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-0.1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-0.1, 1])
    ax.quiver(line1_start[0], line1_start[1], line1_start[2], line1_dir[0], line1_dir[1], line1_dir[2], color='r')
    ax.quiver(line2_start[0], line2_start[1], line2_start[2], line2_dir[0], line2_dir[1], line2_dir[2], color='b')
    ax.scatter(mean_pt[0], mean_pt[1], mean_pt[2], color='g')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.text(line1_start[0], line1_start[1], line1_start[2], 'line1 start')
    ax.text(line2_start[0], line2_start[1], line2_start[2], 'line2 start')
    ax.text(mean_pt[0], mean_pt[1], mean_pt[2], 'intersection')
    plt.show()

