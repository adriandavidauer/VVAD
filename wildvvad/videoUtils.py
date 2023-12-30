#!/usr/bin/env python3
import os
import pathlib
import pickle
import sys

import multiprocessing

from progress.bar import Bar

import numpy as np

from sample import Sample


class videoUtils:
    def __init__(self):
        self.sample = Sample()

    def convert_mp4_to_pickle(self, path: str):
        """
        Method to convert all samples in a folder from mp4 to pickle

        Args:
             path(str): Path containing two sub folders (silent and speaking videos) of wildVVAD dataset
        """

        folders = list(os.walk(path, followlinks=True))[0][1]

        # Create folders for pickle files if they not exist

        if not os.path.exists(os.path.join(path, "faceImages", "positives")):
            # Create a new directory because it does not exist
            os.makedirs(os.path.join(path, "faceImages", "positives"))
            print("Directory for positive samples is created!")
        if not os.path.exists(os.path.join(path, "faceImages", "negatives")):
            # Create a new directory because it does not exist
            os.makedirs(os.path.join(path, "faceImages", "negatives"))
        if not os.path.exists(os.path.join(path, "faceFeatures", "positives")):
            # Create a new directory because it does not exist
            os.makedirs(os.path.join(path, "faceFeatures", "positives"))
            print("Directory for positive samples is created!")
        if not os.path.exists(os.path.join(path, "faceFeatures", "negatives")):
            # Create a new directory because it does not exist
            os.makedirs(os.path.join(path, "faceFeatures", "negatives"))

        folders.sort()

        samples_without_face = []

        print(f"Available folders: {folders}")
        for folder in folders:
            if folder == "silent_videos" or "speaking_videos":
                print(f"Entering folder {folder}")
                current_folder = os.path.abspath(os.path.join(path, folder))
                # open folder and get a list of files
                try:
                    files = list(os.walk(current_folder, followlinks=True))[0][2]
                except:
                    raise Exception(
                        "Data folder is probably not mounted. Or you gave the wrong path.")
                files = [pathlib.Path(os.path.join(current_folder, file))
                         for file in files]
                # get the RefField
                with Bar(f'Converting in {folder}', fill='@', suffix='%(percent).1f%% - %(eta)ds') as bar:
                    for index, file in enumerate(files, start=1):
                        print(f"Current file is {file.name}.")
                        print(f"Processing file number {index} of {len(files)}, "
                              f"== {round((index / len(files) * 100), 2)} in {folder}")
                        # Convert file from avi to mp4
                        command = f"ffmpeg -y -i {file} -vcodec libx264 -crf 28 -r 25 {os.path.join(file.parents[0], file.stem + '.mp4')}"

                        new_file = os.path.join(file.parents[0], file.stem + '.mp4')

                        print(command)
                        os.system(command)

                        current_sample = self.sample.load_video_sample_from_disk(
                            os.path.abspath(new_file))
                        video_sample_face_image = []
                        video_sample_face_feature = []
                        face_detected = True
                        for image in current_sample:
                            # face feature
                            try:
                                print("Get landmarks")
                                preds = self.sample.get_face_landmark_from_sample(image)[-1]
                                # calculate euclidean distance and normalize
                                # outmost eye corner is landmark 36 (right eye) and landmark 45 (left eye)
                                # get euclidean distance
                                corner_right_eye = preds[36]
                                corner_left_eye = preds[45]
                                euclidean_distance = np.linalg.norm(corner_left_eye - corner_right_eye)
                                # normalize on euclidean distance
                                for i in range(len(preds)):
                                    preds[i] = (1 / euclidean_distance) * preds[i]
                                # self.sample.visualize_3d_landmarks(image, preds, False)
                                print("Align face")
                                rotated_landmarks = self.sample.align_3d_face(preds)
                                # self.sample.visualize_3d_landmarks(image, rotated_landmarks, True)

                                video_sample_face_feature.append(rotated_landmarks)

                                # face images
                                image = np.array(image)
                                video_sample_face_image.append(image)
                            except Exception as e:
                                face_detected = False
                                samples_without_face.append(file)
                                print("Samples without face: ", samples_without_face)
                                print("Error in detecting faces as ", e)
                                continue

                        # save as sample in dictionary
                        sample_dict_face_images = {
                            "data": video_sample_face_image,
                            "label": 0 if folder == "silent_videos" else 1,
                            "featureType": "faceImages"
                        }
                        sample_dict_face_features = {
                            "data": video_sample_face_feature,
                            "label": 0 if folder == "silent_videos" else 1,
                            "featureType": "faceFeatures"
                        }
                        # Save as pickle file
                        if folder == "speaking_videos":
                            with open(os.path.join(path, "faceImages", "positives", str(file.stem) + ".pickle"),
                                      'wb') as pickle_file:
                                pickle.dump(video_sample_face_image, file=pickle_file)
                            if face_detected:
                                with open(os.path.join(path, "faceFeatures", "positives", str(file.stem) + ".pickle"),
                                          'wb') as pickle_file:
                                    pickle.dump(video_sample_face_feature, file=pickle_file)
                        elif folder == "silent_videos":
                            with open(os.path.join(path, "faceImages", "negatives", str(file.stem) + ".pickle"),
                                      'wb') as pickle_file:
                                pickle.dump(video_sample_face_image, file=pickle_file)
                            if face_detected:
                                with open(os.path.join(path, "faceFeatures", "negatives", str(file.stem) + ".pickle"),
                                          'wb') as pickle_file:
                                    pickle.dump(video_sample_face_feature, file=pickle_file)
                        bar.next()
        print(f"Samples without faces ", samples_without_face)


if __name__ == "__main__":
    print(sys.executable)
    print(os.getcwd())
    util = videoUtils()
    util.convert_mp4_to_pickle(path="../../wildvvad_dataset")
