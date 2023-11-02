import os
import pickle

import numpy as np

from wildvvad.sample import Sample


class dataSet:
    def __init__(self):
        self.sample = Sample()

    def create_vector_dataset_from_videos(self, path: str = './utils') -> bool:
        """
        Preprocesses the video files and creates data set for the model.
        The data set consist of pickle files (list objects). Each represents one sample.
        Given the path to the video folders

        Args:
            path (str) : path to folders (pos, neg). Folders name must be 'speaking' and
                        'silent'

        Returns:
            ok (bool): Returns result of data creation (True = Ok, False = Error)
        """

        folders = ['speaking_videos', 'silent_videos']

        subfolders = [f.name for f in os.scandir(path) if f.is_dir()]

        # look for folder existence
        for folder in folders:
            if folder not in subfolders:
                print(f"Folder {folder} with videos not found! Function terminated.")
                return

        # go through each file in folder
        for folder in folders:
            for filename in os.scandir(os.path.join(path, folder)):
                if filename.is_file():
                    # convert to list of landmarks with face forward
                    current_sample = self.sample.load_sample_from_disk(
                            path=os.path.join(path.folder.filename))

                    video_sample = []
                    idx = 0

                    for image in current_sample:
                        preds = self.sample.get_face_landmark_from_sample(image)[-1]
                        # calculate euclidean distance and normalize
                        # outmost eye corner is landmark 36 (right eye) and
                        # landmark 45 (left eye)
                        # get euclidean distance
                        corner_right_eye = preds[36]
                        corner_left_eye = preds[45]
                        euclidean_distance = np.linalg.norm(
                            corner_left_eye - corner_right_eye)
                        # normalize on euclidean distance
                        for i in range(len(preds)):
                            preds[i] = (1 / euclidean_distance) * preds[i]
                        # self.sample.visualize_3d_landmarks(image, None, False)
                        rotated_landmarks = self.sample.align_3d_face(preds)
                        # handler.visualize_3d_landmarks(image, rotated_landmarks, True)

                        video_sample.append(rotated_landmarks)

                    # save as pickle file
                    with open(filename + '.pickle', 'wb') as handle:
                        pickle.dump(video_sample, handle,
                                    protocol=pickle.HIGHEST_PROTOCOL)
