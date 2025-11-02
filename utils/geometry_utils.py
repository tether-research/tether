import numpy as np
from scipy.spatial.transform import Rotation as R


### Conversions ###
def quat_to_euler(quat, degrees=False):
    euler = R.from_quat(quat).as_euler("xyz", degrees=degrees)
    return euler


def euler_to_quat(euler, degrees=False):
    return R.from_euler("xyz", euler, degrees=degrees).as_quat()


def rmat_to_euler(rot_mat, degrees=False):
    euler = R.from_matrix(rot_mat).as_euler("xyz", degrees=degrees)
    return euler


def euler_to_rmat(euler, degrees=False):
    return R.from_euler("xyz", euler, degrees=degrees).as_matrix()


def rmat_to_quat(rot_mat, degrees=False):
    quat = R.from_matrix(rot_mat).as_quat()
    return quat


def quat_to_rmat(quat, degrees=False):
    return R.from_quat(quat, degrees=degrees).as_matrix()


def rotvec_to_euler(rot_vec, degrees=False):
    return R.from_rotvec(rot_vec).as_euler("xyz", degrees=degrees)


def euler_to_rotvec(euler, degrees=False):
    return R.from_euler("xyz", euler, degrees=degrees).as_rotvec()


def rot6d_to_rmat(rot_6d):
    a1 = rot_6d[..., :3]
    a2 = rot_6d[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=-2)  # (..., 3, 3)


def rmat_to_rot6d(rot_mat):
    return rot_mat[..., :2, :].reshape(*rot_mat.shape[:-2], 6)


def rot6d_to_euler(rot_6d, convention="xyz", degrees=False):
    rot_mat = rot6d_to_rmat(rot_6d)
    return R.from_matrix(rot_mat).as_euler(convention, degrees=degrees)


def euler_to_rot6d(euler_angles, convention="xyz", degrees=False):
    rot_mat = R.from_euler(convention, euler_angles, degrees=degrees).as_matrix()
    return rmat_to_rot6d(rot_mat)


### Operations ###
def quat_diff(target, source):
    result = R.from_quat(target) * R.from_quat(source).inv()
    return result.as_quat()


def angle_diff(target, source, degrees=False):
    target_rot = R.from_euler("xyz", target, degrees=degrees)
    source_rot = R.from_euler("xyz", source, degrees=degrees)
    result = target_rot * source_rot.inv()
    return result.as_euler("xyz")


def pose_diff(target, source, degrees=False):
    lin_diff = np.array(target[..., :3]) - np.array(source[..., :3])
    rot_diff = angle_diff(target[..., 3:6], source[..., 3:6], degrees=degrees)
    result = np.concatenate([lin_diff, rot_diff], axis=-1)
    return result


def add_quats(delta, source):
    result = R.from_quat(delta) * R.from_quat(source)
    return result.as_quat()


def add_angles(delta, source, degrees=False):
    delta_rot = R.from_euler("xyz", delta, degrees=degrees)
    source_rot = R.from_euler("xyz", source, degrees=degrees)
    new_rot = delta_rot * source_rot
    return new_rot.as_euler("xyz", degrees=degrees)


def add_poses(delta, source, degrees=False):
    lin_sum = np.array(delta[..., :3]) + np.array(source[..., :3])
    rot_sum = add_angles(delta[..., 3:6], source[..., 3:6], degrees=degrees)
    result = np.concatenate([lin_sum, rot_sum], axis=-1)
    return result


### Transforms ###
def change_pose_frame(pose, frame, degrees=False):
    R_frame = euler_to_rmat(frame[3:6], degrees=degrees)
    R_pose = euler_to_rmat(pose[3:6], degrees=degrees)
    t_frame, t_pose = frame[:3], pose[:3]
    euler_new = rmat_to_euler(R_frame @ R_pose, degrees=degrees)
    t_new = R_frame @ t_pose + t_frame
    result = np.concatenate([t_new, euler_new])
    return result


def compose_transformation_matrix(pos, rot):
    T = np.eye(4) if len(pos.shape) == 1 else np.tile(np.eye(4), (pos.shape[0], 1, 1))
    T[..., :3, :3] = euler_to_rmat(rot)
    T[..., :3, 3] = pos
    return T


def decompose_transformation_matrix(T):
    pos = T[..., :3, 3]
    rot = rmat_to_euler(T[..., :3, :3])
    return pos, rot


def transform_world_to_camera(pos, rot, extrinsics):
    T_world = compose_transformation_matrix(pos, rot)
    T_camera = extrinsics @ T_world
    return decompose_transformation_matrix(T_camera)


def transform_trajectory_world_to_camera(trajectory, camera_extrinsics):
    if len(trajectory.shape) == 1:
        return transform_trajectory_world_to_camera(trajectory[None, :], camera_extrinsics)[0]

    camera_frame_trajectory = []
    for i in trajectory:
        pos_world, rot_world, gripper_state = i[:3], i[3:6], i[6:7]
        pos_cam, rot_cam = transform_world_to_camera(pos_world, rot_world, camera_extrinsics)
        camera_frame_trajectory.append(np.concatenate([pos_cam, rot_cam, gripper_state]))
    return np.array(camera_frame_trajectory, dtype=trajectory.dtype)


def project_camera_to_image(pos, intrinsics):
    pos = pos.reshape(-1, 3)
    pixel = (intrinsics @ pos.T).T
    pixel = pixel / pixel[:, [2]]
    return pixel[:, :2] if len(pixel) > 1 else pixel[0, :2]


def project_world_coord_to_image(pos, intrinsics, extrinsics):
    rot = np.zeros(3) 
    pos_cam, _= transform_world_to_camera(pos, rot, extrinsics)
    pixel = project_camera_to_image(pos_cam, intrinsics)
    return pixel


def transform_to_gripper_frame(trajectory, gripper_pose):
    trajectory = trajectory.copy()
    T_AW = compose_transformation_matrix(gripper_pose[..., :3], gripper_pose[..., 3:6])
    T_BW = compose_transformation_matrix(trajectory[..., :3], trajectory[..., 3:6])
    T_BA = np.linalg.inv(T_AW) @ T_BW
    trajectory[..., :3], trajectory[..., 3:6] = decompose_transformation_matrix(T_BA)
    return trajectory


def transform_from_gripper_frame(trajectory, gripper_pose):
    trajectory = trajectory.copy()
    T_AW = compose_transformation_matrix(gripper_pose[..., :3], gripper_pose[..., 3:6])
    T_BA = compose_transformation_matrix(trajectory[..., :3], trajectory[..., 3:6])
    T_BW = T_AW @ T_BA
    trajectory[..., :3], trajectory[..., 3:6] = decompose_transformation_matrix(T_BW)
    return trajectory


def camera_center_in_world(R, t):
    return -R.T @ t


def backproject_pixel_to_world_ray(u, v, K, R, t):
    K_inv = np.linalg.inv(K)
    pixel_hom = np.array([u, v, 1.0], dtype=np.float64)
    dir_cam = K_inv @ pixel_hom
    dir_world = R.T @ dir_cam
    dir_world = dir_world / np.linalg.norm(dir_world)
    C = camera_center_in_world(R, t)
    return C, dir_world


def project_3d_direction_to_2d_direction(dir_3d, K, R):
    dir_cam = R @ dir_3d
    pixel_h = K @ dir_cam
    dir_2d = pixel_h[:2] 
    dir_2d = dir_2d / np.linalg.norm(dir_2d)
    return dir_2d[:2]


def closest_points_between_two_lines(p1, d1, p2, d2):
    p1 = np.asarray(p1, dtype=np.float64)
    d1 = np.asarray(d1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    d2 = np.asarray(d2, dtype=np.float64)

    p12 = p1 - p2
    d1d1 = d1.dot(d1)
    d2d2 = d2.dot(d2)
    d1d2 = d1.dot(d2)
    p12d1 = p12.dot(d1)
    p12d2 = p12.dot(d2)

    A = np.array([[ d1d1, -d1d2],
                  [-d1d2,  d2d2]], dtype=np.float64)
    b = np.array([-p12d1, p12d2], dtype=np.float64)
    sol = np.linalg.inv(A) @ b
    s, t = sol[0], sol[1]

    closest_point_line1 = p1 + s * d1
    closest_point_line2 = p2 + t * d2
    return closest_point_line1, closest_point_line2


def distance_point_to_ray(point, ray_origin, ray_direction):
    point = np.asarray(point)
    ray_origin = np.asarray(ray_origin)
    ray_direction = np.asarray(ray_direction)
    ray_direction = ray_direction / np.linalg.norm(ray_direction)
    v = point - ray_origin
    t = np.dot(v, ray_direction)
    closest_point = ray_origin + t * ray_direction
    distance = np.linalg.norm(point - closest_point)
    return distance, closest_point


def distance_point_to_pixel_ray(point_3d, pixel_uv, K, extrinsics):
    R_cam = extrinsics[:3, :3]
    t_cam = extrinsics[:3, 3]
    ray_origin, ray_direction = backproject_pixel_to_world_ray(
        pixel_uv[0], pixel_uv[1], K, R_cam, t_cam
    )
    distance, closest_point = distance_point_to_ray(
        point_3d, ray_origin, ray_direction
    )
    return distance, closest_point


def triangulate_two_rays(u1, v1, K1, R1, t1,
                         u2, v2, K2, R2, t2):
    p1, d1 = backproject_pixel_to_world_ray(u1, v1, K1, R1, t1)
    p2, d2 = backproject_pixel_to_world_ray(u2, v2, K2, R2, t2)
    c1, c2 = closest_points_between_two_lines(p1, d1, p2, d2)
    point_3D = 0.5 * (c1 + c2)
    return point_3D


def compute_3dray_projection_in_another_image(pixel_in_cam1, camera_intrinsics1, camera_extrinsics1, camera_intrinsics2, camera_extrinsics2):
    R1 = camera_extrinsics1[:3, :3]
    t1 = camera_extrinsics1[:3, 3]
    R2 = camera_extrinsics2[:3, :3]
    t2 = camera_extrinsics2[:3, 3]

    camera1_center_in_world, ray_direction_in_world = backproject_pixel_to_world_ray(
        pixel_in_cam1[0], pixel_in_cam1[1], camera_intrinsics1, R1, t1
    )
    cam1_center_in_image2 = project_world_coord_to_image(
        camera1_center_in_world, camera_intrinsics2, camera_extrinsics2
    )
    sampled_pt_along_ray = camera1_center_in_world + ray_direction_in_world * 2
    sampled_pt_along_ray_in_image2 = project_world_coord_to_image(
        sampled_pt_along_ray, camera_intrinsics2, camera_extrinsics2
    )
    direction_in_image2 = sampled_pt_along_ray_in_image2 - cam1_center_in_image2
    direction_in_image2 /= np.linalg.norm(direction_in_image2)
    return camera1_center_in_world, ray_direction_in_world, cam1_center_in_image2, direction_in_image2


def project_point_to_line(point, line_point_1, line_point_2):
    P, A, B = np.array(point), np.array(line_point_1), np.array(line_point_2)
    AB, AP = B - A, P - A
    return A + np.dot(AP, AB) / np.dot(AB, AB) * AB


def compute_stereo_triangulation(point1, point2, camera_intrinsics1, camera_extrinsics1, camera_intrinsics2, camera_extrinsics2):
    camera1_center_in_world, pixel1_ray_direction_in_world, cam1_center_in_image2 , direction_in_image2 = compute_3dray_projection_in_another_image(point1, camera_intrinsics1, camera_extrinsics1, camera_intrinsics2, camera_extrinsics2)
    assert direction_in_image2[0] > 0, "Direction in image2 should be positive"
    camera2_center_in_world, pixel2_ray_direction_in_world, cam2_center_in_image1, direction_in_image1 = compute_3dray_projection_in_another_image(point2, camera_intrinsics2, camera_extrinsics2, camera_intrinsics1, camera_extrinsics1)
    assert direction_in_image1[0] < 0, "Direction in image1 should be negative"
    point_3d = closest_points_between_two_lines(camera1_center_in_world, pixel1_ray_direction_in_world, camera2_center_in_world, pixel2_ray_direction_in_world)
    dist_between_two_lines = np.linalg.norm(point_3d[0] - point_3d[1])
    mean_point_3d = np.mean([point_3d[0], point_3d[1]], axis=0)

    infos = {
        "cam1_center_in_image2": cam1_center_in_image2,
        "cam2_center_in_image1": cam2_center_in_image1,
        "direction_in_image1": direction_in_image1,
        "direction_in_image2": direction_in_image2
    }
    return mean_point_3d, dist_between_two_lines, infos

