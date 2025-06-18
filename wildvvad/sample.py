import collections
import os
import glob
import pickle

import cv2
import numpy as np
import face_alignment
from matplotlib import pyplot as plt
from utils import utils
from utils import faceFeatureUtils

import open3d as o3d

class Sample:
    def __init__(self):
        pass

    @staticmethod
    def load_sample_objects_from_disk(folder_path: str):
        """
        Loads all sample objects (pickle files) from a folder

        Args:
            folder_path (str): path to sample file
        Returns:
            video_samples_objects (): Generator with sample objects
        """

        old_cwd = os.getcwd()
        os.chdir(folder_path)
        for file in glob.glob("*.pickle"):
            with open(file, 'rb') as pickle_file:
                yield pickle.load(pickle_file)

        # Reset to previous working directory
        os.chdir(old_cwd)

    @staticmethod
    def load_video_sample_from_disk(file_path: str):
        """
        Loads all video samples from a specified folder.

        Args:
            file_path (str): path to sample file
        Returns:
            video_samples (): Generator with frames of one sample
        """

        # with os.scandir(path) as folder:
        # for file in path:  # folder:
        count = 0
        # ToDo check for file type
        # if file.name.endswith(".XXX"):
        video_path = os.path.join(file_path)
        vid_obj = cv2.VideoCapture(video_path)

        if not vid_obj.isOpened():
            print("could not open :", video_path)
            return

        frame_num = int(vid_obj.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_fps = vid_obj.get(cv2.CAP_PROP_FPS)

        # label = True
        # config = {"fps": vid_fps}
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

    @staticmethod
    def get_face_landmark_from_sample(image):
        """
        Getting predicted face landmarks from a sample's frame image

        Args:
            image (cv2 img): Image to detect a face and predict landmarks
        Returns:
            landmarks (array): array of 68 three-dimensional facial landmarks

        """
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D,
                                          flip_input=False, device='cpu')

        return fa.get_landmarks(image)

    def visualize_3d_landmarks(self, image, landmarks, landmarks_available):
        """
        Method to plot the calculated 3 dimensional landmarks. It will plot the image
        next to the detected landmarks for a direct comparison.

        Args:
            image (cv2): Image data to compare to calculated landmarks
            landmarks (numpy array): Calculated landmarks from the provided image
            landmarks_test (bool): If false: generate landmarks from image within this
                method
        """
        if not landmarks_available:
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

        for pred_type in pred_types.values():
            ax.plot3D(preds[pred_type.slice, 0] * 1.2,
                      preds[pred_type.slice, 1],
                      preds[pred_type.slice, 2], color='blue')

        ax.view_init(elev=90., azim=90.)
        ax.set_xlim(ax.get_xlim()[::-1])
        plt.show()

    def align_3d_face(self, landmarks_prediction):
        """
        Align 3 dimensional landmarks so that the face is aligned to the x-y-plane

        Args:
            landmarks_prediction (numpy array): Calculated landmarks from an image
        Returns:
            aligned_landmarks (numpy array): Altered landmarks with aligned facial
                orientation
        """
        # extract the left and right eye (x, y)-coordinates
        (l_start, l_end) = utils.FACIAL_LANDMARKS["left_eye"]
        (r_start, r_end) = utils.FACIAL_LANDMARKS["right_eye"]

        left_eye_pts = landmarks_prediction[l_start:l_end]
        right_eye_pts = landmarks_prediction[r_start:r_end]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(landmarks_prediction)

        # compute center of mass for each eye column wise
        left_eye_center = left_eye_pts.mean(axis=0).astype("float")
        right_eye_center = right_eye_pts.mean(axis=0).astype("float")

        # compute the angle between the eye centroids
        dX = right_eye_center[0] - left_eye_center[0]
        dY = right_eye_center[1] - left_eye_center[1]
        dZ = right_eye_center[2] - left_eye_center[2]

        # First, rotate in X-Z
        # Get Angles
        angle_x = np.degrees(np.arctan2(dZ, dY)) - 90
        angle_y = np.degrees(np.arctan2(dZ, dX)) - 180
        angle_z = np.degrees(np.arctan2(dY, dX)) - 180

        # Rotation Matrix 3x3
        R = pcd.get_rotation_matrix_from_xyz((np.deg2rad(angle_x), np.deg2rad(angle_y),
                                              np.deg2rad(angle_z)))
        # print(f"Rotation Matrix is {R}")
        center_x = (left_eye_center[0] + right_eye_center[0]) // 2
        center_y = (left_eye_center[1] + right_eye_center[1]) // 2
        center_z = (left_eye_center[2] + right_eye_center[2]) // 2
        pcd = pcd.rotate(R, center=(center_x, center_y, center_z))

        first_rotation_state = np.asarray(pcd.points)

        # Go into second rotation to align wrongly oriented faces
        left_eye_pts = first_rotation_state[l_start:l_end]
        right_eye_pts = first_rotation_state[r_start:r_end]

        # compute center of mass for each eye column wise
        left_eye_center = left_eye_pts.mean(axis=0).astype("float")
        right_eye_center = right_eye_pts.mean(axis=0).astype("float")

        point_cloud = faceFeatureUtils.rotate_face_landmarks(first_rotation_state)
        aligned_point_cloud = faceFeatureUtils.orient_face_landmarks(point_cloud)

        shifted_point_cloud = utils.shift_to_positive_range(aligned_point_cloud)

        return shifted_point_cloud

# Ressources:

# https://pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
# https://stackoverflow.com/questions/47475976/face-alignment-in-video-using-python
# https://medium.com/@dsfellow/precise-face-alignment-with-opencv-dlib-e6c8acead262


# https://medium.com/@rdadlaney/basics-of-3d-point-cloud-data-manipulation-in-python-95b0a1e1941e
