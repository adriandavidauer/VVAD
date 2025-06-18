import numpy as np
from collections import OrderedDict

# For dlib’s 68-point facial landmark detector:
FACIAL_LANDMARKS_68 = OrderedDict([
    ("mouth", (48, 68)),
    ("inner_mouth", (60, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])

# in order to support legacy code, we'll default the indexes to the
# 68-point model
FACIAL_LANDMARKS = FACIAL_LANDMARKS_68


def translate_point_cloud(point_cloud, translation_vector):
    """
    Complete multidimensional point cloud is translated by a fixed vector

    Args:
        point_cloud(np array): multidimensional point cloud as numpy array
        translation_vector(np array): translation vector, dimension must fit
    Returns:
        translated_point_cloud (np array): translated point cloud as np array

    """
    return point_cloud - translation_vector


def rotate_point_cloud_matrix(point_cloud, rotation_matrix):
    """
    Rotates a multidimensional point cloud around a rotation matrix T

    Args:
        point_cloud(np array): multidimensional point cloud as numpy array
        rotation_matrix(np array): rotation matrix, dimension must fit
    Returns:
        rotated_point_cloud (np array): Rotated point cloud as np array

    """
    return np.dot(point_cloud, rotation_matrix.T)


def compute_centroid(points):
    """
    Computes controid of multiple points

    Args:
        points(np array): point cloud
    Returns:
        centroid(np array): position values of the centroid
    """
    return np.mean(points, axis=0)


def compute_rotation_angle(nose_bridge_landmarks):
    """
    Computes the rotation angle of the nose bridge of facial landmarks to the z axis

    Args:
        nose_bridge_landmarks (np array): position vectors of the nose bridge landmarks
    Returns:
        angle (float): Angle between node bridge and horizontal axis

    """
    # Compute the vector between the two nose bridge landmarks
    vector = nose_bridge_landmarks[1] - nose_bridge_landmarks[0]

    # Compute the angle between the vector and the horizontal axis
    angle = np.arctan2(vector[1], vector[0])

    return angle


def rotate_point_cloud_angle(points, angle):
    """
    Rotate point cloud around an angle in x-y-plane

    Args:
        points (np array): three dimensional point cloud of landmarks
        angle (float): Angle to rotate
    Returns:
        rotated_points (np array): point cloud rotated around angle in x-y-plane

    """
    # Define rotation matrix
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                [np.sin(angle), np.cos(angle), 0],
                                [0, 0, 1]])

    # Rotate the points
    rotated_points = np.dot(points, rotation_matrix.T)

    return rotated_points


def orient_face_landmarks(face_landmarks):  # pragma: no cover
    """
    The raw facial landmarks will be oriented in a way that the face should be shown
    from the front on the x-y-plane as good as possible

    Args:
        face_landmarks (np array): three dimensional facial landmarks as predicted
    Returns:
        aligned_landmarks (np array): Aligned three dimensional facial landmarks

    """
    # Flatten the face landmarks into a single array
    flat_landmarks = face_landmarks.reshape(-1, 3)

    # Compute centroid of the face landmarks
    centroid = compute_centroid(flat_landmarks)

    # Center the points around the centroid
    centered_points = flat_landmarks - centroid

    # Extract nose bridge landmarks
    nose_bridge_landmarks = centered_points[27:31,
                            :2]  # Only considering X and Y coordinates

    # Compute rotation angle
    rotation_angle = compute_rotation_angle(nose_bridge_landmarks)

    # Adjust rotation angle by 90 degrees
    rotation_angle -= np.pi / 2  # Subtract 90 degrees in radians

    # Rotate the face landmarks around the z-axis
    rotated_points = rotate_point_cloud_angle(centered_points, rotation_angle)

    # Reshape the rotated points back to the original shape
    rotated_face_landmarks = rotated_points + centroid
    rotated_face_landmarks = rotated_face_landmarks.reshape(face_landmarks.shape)

    return rotated_face_landmarks


def rotate_face_landmarks(face_landmarks):  # pragma: no cover
    """
    Rotate face landmarks
        Args:
            face_landmarks (np array): three-dimensional facial landmarks as predicted
    Returns:
        rotated_landmarks (np array): Rotated three-dimensional facial landmarks

    """
    # Flatten the face landmarks into a single array
    flat_landmarks = face_landmarks.reshape(-1, 3)

    # Compute centroid of the face landmarks
    centroid = np.mean(flat_landmarks, axis=0)

    # Center the points around the centroid
    centered_points = flat_landmarks - centroid

    # Compute the covariance matrix
    covariance_matrix = np.cov(centered_points.T)

    # Perform SVD to find the rotation matrix
    _, _, Vt = np.linalg.svd(covariance_matrix)
    rotation_matrix = Vt.T

    # Rotate the point cloud
    rotated_points = np.dot(centered_points, rotation_matrix)

    # Reshape the rotated points back to the original shape
    rotated_face_landmarks = rotated_points + centroid
    rotated_face_landmarks = rotated_face_landmarks.reshape(face_landmarks.shape)

    return rotated_face_landmarks


def calculate_rotation_matrix(axis, theta):
    """
    Calculated rotation matrix from an axis and an angle

    Args:
        axis: rotation axis
        theta: rotation angle
    Returns:
        rotation_matrix: Resulting rotation matrix from axis and angle theta
    """
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    rotation_matrix = np.array(
        [[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (a * c + b * d)],
         [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
         [2 * (b * d - a * c), 2 * (a * b + c * d), a * a - b * b - c * c + d * d]])
    return rotation_matrix


def align_face_to_z_axis(point_cloud):
    """
    Alignes a three dimensional face to the z axis of a cartesian space

    Args:
        point_cloud (np array): three dimensional face landmarks
    Returns:
        aligned_point_cloud (np array): Newly aligned face landmarks to the z axis
    """
    # Find the indices for the right and left eye corners
    right_eye_index = 36
    left_eye_index = 45

    # Step 1: Translate the point cloud
    middle_points_between_eyes = (point_cloud[right_eye_index] + point_cloud[
        left_eye_index]) / 2.0
    translated_point_cloud = translate_point_cloud(point_cloud,
                                                   middle_points_between_eyes)

    # Step 2: Calculate rotation axis and angle
    right_eye_target = np.array([-0.5, 0, 0])
    left_eye_target = np.array([0.5, 0, 0])

    right_eye_current = translated_point_cloud[right_eye_index]
    left_eye_current = translated_point_cloud[left_eye_index]

    rotation_axis = np.cross(right_eye_target - left_eye_target,
                             right_eye_current - left_eye_current)
    rotation_angle = np.arccos(np.dot(right_eye_target - left_eye_target,
                                      right_eye_current - left_eye_current) / (
                                       np.linalg.norm(
                                           right_eye_target - left_eye_target) * np.linalg.norm(
                                   right_eye_current - left_eye_current)))

    # Step 3: Calculate rotation matrix
    rotation_matrix = calculate_rotation_matrix(rotation_axis, rotation_angle)

    # Step 4: Apply rotation to the point cloud
    aligned_point_cloud = rotate_point_cloud_matrix(translated_point_cloud,
                                                    rotation_matrix)

    # Step 5: Calculate rotation axis and angle to chin
    chin_current_dist = np.sqrt(aligned_point_cloud[8][0] ** 2 +
                                aligned_point_cloud[8][1] ** 2 +
                                aligned_point_cloud[8][2] ** 2)
    chin_current = aligned_point_cloud[8]
    chin_target = np.array([0, -chin_current_dist, 0])

    rotation_axis = np.cross(chin_target, chin_current)
    rotation_angle = np.arccos(np.dot(chin_target, chin_current) / (
            np.linalg.norm(chin_target) * np.linalg.norm(chin_current)
    ))

    # Step 6: Calculate rotation matrix
    rotation_matrix = calculate_rotation_matrix(rotation_axis, rotation_angle)

    # Step 7: Apply rotation to the point cloud
    new_aligned_point_cloud = rotate_point_cloud_matrix(aligned_point_cloud,
                                                        rotation_matrix)

    # Step 8: Inverse all z values

    new_aligned_point_cloud[:, 2] = [-element for element in
                                     new_aligned_point_cloud[:, 2]]

    return new_aligned_point_cloud
