import os
import pickle
import random

import numpy as np

from wildvvad.sample import Sample
from wildvvad.utils.kerasUtils import kerasUtils


class dataSet:
    def __init__(self):
        self.sample = Sample()

    def create_vector_dataset_from_videos(self, path: str = './utils') -> bool:
        """
        Preprocesses the video files and creates data set for the model.
        The data set consist of pickle files (list objects). Each represents one sample.
        Given the path to the video folders

        Args:
            path (str) : path to folders (pos, neg). Folders name must be 'speaking_videos' and
                        'silent_videos'

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
            print(f"Enter folder {folder}")
            for filename in os.scandir(os.path.join(path, folder)):
                if filename.is_file():
                    print(f"Found file {filename}")
                    # convert to list of landmarks with face forward
                    print(f"Get sample from {os.path.join(filename)}")
                    current_sample = self.sample.load_video_sample_from_disk(
                        file_path=os.path.join(filename))

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
                        print("Normalized to euclidean distance.")
                        # self.sample.visualize_3d_landmarks(image, None, False)
                        rotated_landmarks = self.sample.align_3d_face(preds)
                        # handler.visualize_3d_landmarks(image, rotated_landmarks, True)

                        video_sample.append(rotated_landmarks)

                    print(f"Safe file with label. Label "
                          f"is {True if folder == 'speaking_videos' else False}")
                    sample_with_label = {
                        "sample": video_sample,
                        "label": True if folder == "speaking_videos" else False
                    }

                    # save as pickle file
                    save_path = os.path.join(filename)
                    with open(save_path.replace('.mp4', '') + '.pickle',
                              'wb') as handle:
                        pickle.dump(sample_with_label, handle,
                                    protocol=pickle.HIGHEST_PROTOCOL)

    def load_data_set_from_pickles(self, path: str = './utils') -> list:
        """
        Load complete dataset from available sample pickle files

        Args:
            path (str) : path to folders (pos, neg). Folders name must be
                        'speaking_videos' and
                        'silent_videos'

        Returns:
            dataset (list): Returns all data as list of dict (sample, label)
        """

        loaded_dataset = []

        folders = ['speaking_videos', 'silent_videos']

        subfolders = [f.name for f in os.scandir(path) if f.is_dir()]

        # look for folder existence
        for folder in folders:
            if folder not in subfolders:
                print(f"Folder {folder} with videos not found! Function terminated.")
                return

        # go through each file in folder
        for folder in folders:
            print(f"Enter folder {folder}")

            data = self.sample.load_sample_objects_from_disk(
                folder_path=os.path.join(path, folder))

            for datapoint in data:
                loaded_dataset.append(datapoint)

            print(f"Length of dataset is {len(loaded_dataset)}")

        random.seed(42)
        random.shuffle(loaded_dataset)

        return loaded_dataset


if __name__ == '__main__':
    dataset = dataSet()
    # dataset.create_vector_dataset_from_videos()
    # dataset = dataset.load_data_set_from_pickles()
    # kerasUtilities = kerasUtils()
    # kerasUtilities.train_test_split(dataset=dataset)
