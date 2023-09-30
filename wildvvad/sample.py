import collections
import os

import cv2
import numpy as np
import face_alignment
from matplotlib import pyplot as plt


class Sample:
    def __init__(self):
        pass

    def load_samples_from_disk(self, path: str):
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


    def visualize_3d_landmarks(self, image):
        preds = self.get_face_landmark_from_sample(image)[-1]
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


#https://pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
#https://stackoverflow.com/questions/47475976/face-alignment-in-video-using-python
