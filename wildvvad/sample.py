import collections
import os

import cv2
import numpy
import numpy as np
import face_alignment
from matplotlib import pyplot as plt
from utils import utils
import vg

import open3d as o3d


class Sample:
    def __init__(self):
        pass

    def load_sample_from_disk(self, path: str):
        """
        Loads all video samples from a specified folder.

        Args:
            path (str): Path to folder with samples
        Returns:
            video_samples (): Generator with video samples
        """

        with os.scandir(path) as folder:
            for file in folder:
                count = 0
                # ToDo check for file type
                # if file.name.endswith(".XXX"):
                video_path = os.path.join(path, file.name)
                vid_obj = cv2.VideoCapture(video_path)

                if not vid_obj.isOpened():
                    print("could not open :", video_path)
                    return

                frame_num = int(vid_obj.get(cv2.CAP_PROP_FRAME_COUNT))
                vid_fps = vid_obj.get(cv2.CAP_PROP_FPS)

                label = True
                config = {"fps": vid_fps}
                print("FPS are ", vid_fps)

                success = vid_obj.grab()

                if not success:
                    raise Exception(
                        "Couldn't grab frame of file {}".format(video_path))

                # grab frames from start to end frame
                while success:
                    _, image = vid_obj.retrieve()

                    # ToDo needed?
                    if count <= frame_num:
                        pass
                        # data.append(image)
                    count += 1
                    if count > frame_num:
                        break

                    success = vid_obj.grab()

                    yield image

    def get_face_landmark_from_sample(self, image):
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D,
                                          flip_input=False, device='cpu')

        return fa.get_landmarks(image)

    def visualize_3d_landmarks(self, image, landmarks, landmarks_test):
        if not landmarks_test:
            preds = self.get_face_landmark_from_sample(image)[-1]
        else:
            preds = landmarks
        # 2D-Plot
        plot_style = dict(marker='o',
                          markersize=4,
                          linestyle='-',
                          lw=2)

        pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
        pred_types = {'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
                      'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
                      'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
                      'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
                      'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
                      'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
                      'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
                      'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
                      'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
                      }

        fig = plt.figure(figsize=plt.figaspect(.5))
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(image)

        for pred_type in pred_types.values():
            ax.plot(preds[pred_type.slice, 0],
                    preds[pred_type.slice, 1],
                    color=pred_type.color, **plot_style)

        ax.axis('off')

        # 3D-Plot
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        surf = ax.scatter(preds[:, 0] * 1.2,
                          preds[:, 1],
                          preds[:, 2],
                          c='cyan',
                          alpha=1.0,
                          edgecolor='b')

        for pred_type in pred_types.values():
            ax.plot3D(preds[pred_type.slice, 0] * 1.2,
                      preds[pred_type.slice, 1],
                      preds[pred_type.slice, 2], color='blue')

        ax.view_init(elev=90., azim=90.)
        ax.set_xlim(ax.get_xlim()[::-1])
        plt.show()

    def align_3d_face(self, landmarks_prediction):
        # convert landmark (x, y, z) - coordinates to a NumPy array
        # shape = utils.shape_to_np(landmarks_prediction)
        # extract the left and right eye (x, y)-coordinates
        (l_start, l_end) = utils.FACIAL_LANDMARKS["left_eye"]
        (r_start, r_end) = utils.FACIAL_LANDMARKS["right_eye"]

        left_eye_pts = landmarks_prediction[l_start:l_end]
        right_eye_pts = landmarks_prediction[r_start:r_end]

        print(left_eye_pts)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(landmarks_prediction)
        # o3d.io.write_point_cloud("./data.ply", pcd)

        print(f"PCD is {pcd}")
        print(f"Numpy convert is {np.asarray(pcd)}")
        print(type(pcd))
        print(type(np.asarray(pcd)))
        # o3d.visualization.draw_geometries([pcd])

        # compute center of mass for each eye column wise
        left_eye_center = left_eye_pts.mean(axis=0).astype("float")
        right_eye_center = right_eye_pts.mean(axis=0).astype("float")

        print("Center", left_eye_center)

        # compute the angle between the eye centroids
        dX = right_eye_center[0] - left_eye_center[0]
        dY = right_eye_center[1] - left_eye_center[1]
        dZ = right_eye_center[2] - left_eye_center[2]
        vector_angle = vg.angle(right_eye_center, left_eye_center)
        print("Angle is", vector_angle)

        # compute center (x, y, z)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                       (left_eye_center[1] + right_eye_center[1]) // 2,
                       (left_eye_center[2] + right_eye_center[2]) // 2
                       )

        # First, rotate in X-Z
        # Get Angles
        angle_x = np.degrees(np.arctan2(dZ, dY)) - 90
        angle_y = np.degrees(np.arctan2(dZ, dX)) - 180
        angle_z = np.degrees(np.arctan2(dY, dX)) - 180

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        ## desiredRightEyeX = 1.0 - self.desiredLeftEye[0]
        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        ## dist = np.sqrt((dX ** 2) + (dY ** 2))
        ## desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        ## desiredDist *= self.desiredFaceWidth
        ## scale = desiredDist / dist

        eyes_center_x = ((left_eye_center[1] + right_eye_center[1]) // 2,
                         (left_eye_center[2] + right_eye_center[2]) // 2)
        eyes_center_y = ((left_eye_center[0] + right_eye_center[0]) // 2,
                         (left_eye_center[2] + right_eye_center[2]) // 2)
        eyes_center_z = ((left_eye_center[0] + right_eye_center[0]) // 2,
                         (left_eye_center[1] + right_eye_center[1]) // 2)

        # grab the rotation matrix for rotating and scaling the face
        M_x = cv2.getRotationMatrix2D(eyes_center_x, angle_x, 1.0)
        M_y = cv2.getRotationMatrix2D(eyes_center_y, angle_y, 1.0)
        M_z = cv2.getRotationMatrix2D(eyes_center_z, angle_z, 1.0)

        # Rotation Matrix 3x3
        R = pcd.get_rotation_matrix_from_xyz((np.deg2rad(angle_x), np.deg2rad(angle_y),
                                              np.deg2rad(angle_z)))
        print(f"Rotation Matrix is {R}")
        center_x = (left_eye_center[0] + right_eye_center[0]) // 2
        center_y = (left_eye_center[1] + right_eye_center[1]) // 2
        center_z = (left_eye_center[2] + right_eye_center[2]) // 2
        pcd = pcd.rotate(R, center=(center_x, center_y, center_z))
        # o3d.visualization.draw_geometries([pcd])
        o3d.geometry.Geometry
        o3d.utility.Matrix3dVector

        return np.asarray(pcd.points)

        # transformed_point_cloud = rotation_matrix @ point_cloud_array + translation_vector

        # update the translation component of the matrix
        #tX = self.desiredFaceWidth * 0.5
        #tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        #M[0, 2] += (tX - eyes_center_xz[0])
        #M[1, 2] += (tY - eyes_center_xz[1])

        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h),
                                flags=cv2.INTER_CUBIC)

        # transformed_point_cloud = rotation_matrix @ point_cloud_array + translation_vector

        # return the aligned face
        return output

        # Second, rotate in X-Y
        """
        angle_xy = np.degrees(np.arctan2(dY, dX)) - 180
        ...
        """


def angle(v1, v2, acute):
    # v1 is your first vector
    # v2 is your second vector
    calc_angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    if acute:
        return calc_angle
    else:
        return 2 * np.pi - calc_angle

# https://pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
# https://stackoverflow.com/questions/47475976/face-alignment-in-video-using-python
# https://medium.com/@dsfellow/precise-face-alignment-with-opencv-dlib-e6c8acead262


# https://medium.com/@rdadlaney/basics-of-3d-point-cloud-data-manipulation-in-python-95b0a1e1941e

# xyz = np.random.rand(100, 3)
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(xyz)
# o3d.io.write_point_cloud("./data.ply", pcd)

# o3d.visualization.draw_geometries([pcd])
