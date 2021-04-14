"""utils for videos"""

# System imports
import os
import pathlib
import argparse
from multiprocessing import Process
import shutil
from collections import defaultdict
import json
from pathlib import Path


# 3rd party imports
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
import yaml
from dvg_ringbuffer import RingBuffer
from vvadlrs3 import pretrained_models, sample, dlibmodels
from vvadlrs3.utils.imageUtils import cropImage


# local imports


def getFramesfromVideo(video_path):
    """
    yields the frames from a video
    """
    success = True
    vidObj = cv2.VideoCapture(str(video_path))
    while(success):
        success, image = vidObj.read()
        yield success, image


def analyzeVideo(video_path, feature_type='faceImage', save_as_json=None):
    """
    returns a analysis of the video in the following format:

    analysis = {
        video_path: path to the video
        fps: the fps associated with the video
        feature_type: One out of ["faceImage", "lipImage", "faceFeatures", "lipFeatures"]
        frame_scores: dict of lists with the prediction of every frame. (A frame has k predictions if it is not in the beginning or end of the video because a sample has k frames and the samples overlap.)

    }

    :param video_path: path to the video file to analyze
    :type video_path: 
    :param feature_type: type of the features that should be used when creating samples.
    :type feature_type: String ["faceImage", "lipImage", "faceFeatures", "lipFeatures"]
    :param save_as_json: Path where to save the analysis as json file
    :type save_as_json: String
    :returns: analysis dict
    """
    analysis = {'video_path': video_path,
                'feature_type': feature_type}
    if feature_type == 'faceImage':
        model = pretrained_models.getFaceImageModel()  # model for predictions
        k = 36  # Number of frames used for inference
        shape = (96, 96)  # Resolution of the input imgae for the prediction
        featureType = 'faceImage'  # Type of the features that will be created from the Image

    elif feature_type == 'lipImage':
        raise NotImplementedError(
            'Analysis for lipImage is not implemented yet')
    elif feature_type == 'faceFeatures':
        raise NotImplementedError(
            'Analysis for faceFeatures is not implemented yet')
    elif feature_type == 'lipFeatures':
        raise NotImplementedError(
            'Analysis for lipFeatures is not implemented yet')
    else:
        raise ValueError(
            'feature_type must be one of ["faceImage", "lipImage", "faceFeatures", "lipFeatures"]')

    # TODO: this should actually only be needed if not using faceImage type
    shapeModelPath = str(dlibmodels.SHAPE_PREDICTOR_68_FACE_LANDMARKS())
    ffg = sample.FaceFeatureGenerator(
        featureType, shapeModelPath=shapeModelPath, shape=shape)

    # TODO: Fist approach only with a detector - later we can try FaceTracker for multiple faces?
    detector = dlib.get_frontal_face_detector()

    # Ringbuffer for features
    rb = RingBuffer(k, dtype=(np.uint8, (*shape, 3)))
    cap = cv2.VideoCapture(video_path)
    analysis['fps'] = cap.get(cv2.CAP_PROP_FPS)

    i = 0
    frame_scores = defaultdict(list)
    for ret, frame in getFramesfromVideo(video_path):
        dets = detector(frame, 1)   # Detect faces
        if dets:
            features = ffg.getFeatures(cropImage(frame, dets[0]))
            # fill ringbuffer
            rb.append(features)
            if rb.is_full:
                y = model.predict(np.array([rb]))
                frame_scores[i].append(y)
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
