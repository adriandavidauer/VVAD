"""
This Module has all the pretrained Models for the VVAD-LRS3
"""
# System imports
from pathlib import Path

# 3rd party imports
from keras.models import load_model


# local imports

__author__ = 'Adrian Lubitz'

base_path = Path(__file__).absolute().parent


def get_face_img_model():
    """
    returns the pretrained Face Image Model
    """
    return load_model(str(base_path) / "bestFaceEndToEnd.h5")


def get_lip_img_model():
    """
    returns the pretrained Lip Image Model
    """
    return load_model(str(base_path) / "bestLipEndToEnd.h5")


def get_face_feature_model():
    """
    returns the pretrained Face Feature Model
    """
    return load_model(str(base_path) / "faceFeatureModel.h5")


def get_lip_feature_model():
    """
    returns the pretrained Lip Feature Model
    """
    return load_model(str(base_path) / "lipFeatureModel.h5")


def get_face_img_model_path():
    """
    returns the path to pretrained Face Image Model
    """
    return str(base_path) / "bestFaceEndToEnd.h5"


def get_lip_img_model_path():
    """
    returns the path to pretrained Lip Image Model
    """
    return str(base_path) / "bestLipEndToEnd.h5"


def get_face_feature_model_path():
    """
    returns the path to pretrained Face Feature Model
    """
    return str(base_path) / "faceFeatureModel.h5"


def get_lip_feature_model_path():
    """
    returns the path to pretrained Lip Feature Model
    """
    return str(base_path) / "lipFeatureModel.h5"
