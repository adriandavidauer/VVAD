"""utils for videos"""

import json
# System imports
from collections import defaultdict
from pathlib import Path

# 3rd party imports
import cv2
import dlib
import numpy as np
from dvg_ringbuffer import RingBuffer

from vvadlrs3 import pretrained_models, sample, dlibmodels
from vvadlrs3.dataSet import transform_points_to_numpy
from vvadlrs3.utils.imageUtils import crop_img


# local imports


def get_frames_from_video(video_path):
    """ yields the frames from a video

    Args:
        video_path (str): Path to the video file
    """
    success = True
    vid_obj = cv2.VideoCapture(str(video_path))
    while success:
        success, image = vid_obj.read()
        if not success:
            return
        yield success, image


def analyze_video(video_path, feature_type='faceImage', save_as_json=None):
    """
    returns a analysis of the video in the following format:

    analysis = {
        video_path: path to the video
        fps: the fps associated with the video
        feature_type: One out of ["faceImage", "lipImage", "faceFeatures",
         "lipFeatures"]
        frame_scores: dict of lists with the prediction of every frame.
            (A frame has k predictions if it is not in the
            beginning or end of the video because a sample has k frames and the
            samples overlap.)

    }

    Args:
        video_path (str): path to the video file to analyze
        feature_type (str): type of the features that should be used when creating
        samples. ["faceImage", "lipImage", "faceFeatures", "lipFeatures"]
        save_as_json (str): Path where to save the analysis as json file

    Returns:
        analysis (dict):
    """
    analysis = {'video_path': video_path,
                'feature_type': feature_type}
    if feature_type == 'faceImage':
        model = pretrained_models.get_face_img_model()  # model for predictions
    elif feature_type == 'lipImage':
        model = pretrained_models.get_lip_img_model()  # model for predictions
    elif feature_type == 'faceFeatures':
        model = pretrained_models.get_face_feature_model()  # model for predictions
    elif feature_type == 'lipFeatures':
        model = pretrained_models.get_lip_feature_model()  # model for predictions
    else:
        raise ValueError(
            'feature_type must be one of ["faceImage", "lipImage", "faceFeatures", '
            '"lipFeatures"]')

    k = model.layers[0].input_shape[1]  # Number of frames used for inference
    # Type of the features that will be created from the Image
    feature_type = feature_type
    input_shape = model.layers[0].input_shape[2:]
    # TODO: this should actually only be needed if not using faceImage type
    shape_model_path = str(dlibmodels.SHAPE_PREDICTOR_68_FACE_LANDMARKS())
    ffg = sample.FaceFeatureGenerator(
        feature_type, shape_model_path=shape_model_path, shape=(input_shape[1],
                                                                input_shape[0]))

    # TODO: Fist approach only with a detector - later we can try FaceTracker for
    #  multiple faces?
    detector = dlib.get_frontal_face_detector()

    # Ringbuffer for features
    rb = RingBuffer(k, dtype=(np.uint8, input_shape))
    cap = cv2.VideoCapture(video_path)
    analysis['fps'] = cap.get(cv2.CAP_PROP_FPS)

    i = 0
    frame_scores = defaultdict(list)
    for ret, frame in get_frames_from_video(video_path):
        dets = detector(frame, 1)   # Detect faces
        if dets:
            features = ffg.get_features(crop_img(frame, dets[0]))
            if "Features" in feature_type:
                features = transform_points_to_numpy(features)
            # fill ringbuffer
            rb.append(features)
            if rb.is_full:
                y = model.predict(np.array([rb]))
                for x in range(i - (k-1), i):  # append to all involved frames
                    # cast to float64 to make it json serializable
                    frame_scores[x].append(np.float64(y[0, 0]))
        else:
            # empty ringbuffer - to prevent glitches
            rb.clear()
        i += 1
    cap.release()
    analysis['frame_scores'] = frame_scores
    if save_as_json:
        save_as_json = Path(save_as_json)
        save_as_json.parent.mkdir(parents=True, exist_ok=True)
        with open(save_as_json, 'w') as outfile:
            json.dump(analysis, outfile)
    return analysis
