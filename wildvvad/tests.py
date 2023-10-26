import numpy as np

from sample import Sample


def main():
    handler = Sample()
    sample = handler.load_sample_from_disk("./videos")
    mean_euclidean = []

    video_sample = []
    idx = 0
    print(f"Data type of sample is {type(sample)}")
    for image in sample:
        preds = handler.get_face_landmark_from_sample(image)[-1]
        # calculate euclidean distance and normalize
        # outmost eye corner is landmark 36 (right eye) and landmark 45 (left eye)
        # get euclidean distance
        corner_right_eye = preds[36]
        corner_left_eye = preds[45]
        euclidean_distance = np.linalg.norm(corner_left_eye - corner_right_eye)
        # normalize on euclidean distance
        for i in range(len(preds)):
            preds[i] = (1 / euclidean_distance) * preds[i]
        handler.visualize_3d_landmarks(image, None, False)
        rotated_landmarks = handler.align_3d_face(preds)
        handler.visualize_3d_landmarks(image, rotated_landmarks, True)

        video_sample.append(rotated_landmarks)
        idx += 1
        if idx > 5:
            break


if __name__ == "__main__":
    main()
