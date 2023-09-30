import math

import numpy as np
from matplotlib import pyplot as plt

from sample import Sample
import statistics
import collections

import cv2


def main():
    handler = Sample()
    samples = handler.load_samples_from_disk("./videos")
    mean_euclidean = []

    for sample in samples:
        preds = handler.get_face_landmark_from_sample(sample)[-1]
        # calculate euclidean distance and normalize
        # outmost eye corner is landmark 36 (right eye) and landmark 45 (left eye)
        # get euclidean distance
        corner_right_eye = preds[36]
        corner_left_eye = preds[45]
        euclidean_distance = np.linalg.norm(corner_left_eye - corner_right_eye)
        # normalize on euclidean distance
        for i in range(len(preds)):
            preds[i] = (1 / euclidean_distance) * preds[i]



        #print(f"Euclidean distance is: {np.linalg.norm(p2 - p1)}")
        #mean_euclidean.append(round(np.linalg.norm(p2 - p1), 2))

        #if len(mean_euclidean) > 3:
        #    print(f"Euclidean mean is {statistics.mean(mean_euclidean)}")
        #    print(f"STD is {(sum([((x - statistics.mean(mean_euclidean)) ** 2) for x in mean_euclidean]) / len(mean_euclidean)) ** 0.5}")


        # handler.visualize_3d_landmarks(sample)

if __name__ == "__main__":
    main()
