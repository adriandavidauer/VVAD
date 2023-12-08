#!/usr/bin/env python3
import os
import pathlib
import pickle
import sys

from progress.bar import Bar

import numpy as np

from wildvvad.sample import Sample

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

        if not os.path.exists(os.path.join(path, "positives")):
            # Create a new directory because it does not exist
            os.makedirs(os.path.join(path, "positives"))
            print("Directory for positive samples is created!")
        if not os.path.exists(os.path.join(path, "negatives")):
            # Create a new directory because it does not exist
            os.makedirs(os.path.join(path, "negatives"))

        folders.sort()

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
                    for file in files:
                        print(f"Current file is {file.name}.")
                        # Convert file from avi to mp4
                        command = f"ffmpeg -y -i {file} -vcodec libx264 -crf 28 -r 25 {os.path.join(file.parents[0], file.stem + '.mp4')}"

                        new_file = os.path.join(file.parents[0], file.stem + '.mp4')

                        print(command)
                        os.system(command)

                        current_sample = self.sample.load_video_sample_from_disk(
                            os.path.abspath(new_file))
                        video_sample = []
                        for image in current_sample:
                            image = np.array(image)
                            video_sample.append(image)

                        # save as sample in dictionary
                        sample_dict = {
                            "data": video_sample,
                            "label": 0 if folder == "silent_videos" else 1,
                            "featureType": "faceImages"
                        }
                        # Save as pickle file
                        if folder == "speaking_videos":
                            with open(os.path.join(path, "positives", str(file.stem) + ".pickle"), 'wb') as pickle_file:
                                pickle.dump(video_sample, file=pickle_file)
                        elif folder == "silent_videos":
                            with open(os.path.join(path, "negatives", str(file.stem) + ".pickle"), 'wb') as pickle_file:
                                pickle.dump(video_sample, file=pickle_file)
                        bar.next()


if __name__ == "__main__":
    util = videoUtils()
    util.convert_mp4_to_pickle(path="../../wildvvad_dataset")
