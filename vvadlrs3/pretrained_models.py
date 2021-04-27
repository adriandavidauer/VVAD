'''
This Module has all the pretrained Models for the VVAD-LRS3
'''
# System imports
from pathlib import Path

# 3rd party imports
from keras.models import load_model


# local imports

__author__ = 'Adrian Lubitz'

base_path = Path(__file__).absolute().parent


def getFaceImageModel():
    '''
    returns the pretrained Face Image Model
    '''
    return load_model(base_path / "bestFaceEndToEnd.h5")


def getLipImageModel():
    '''
    returns the pretrained Lip Image Model
    '''
    return load_model(base_path / "bestLipEndToEnd.h5")


def getFaceFeatureModel():
    '''
    returns the pretrained Face Feature Model
    '''
    return load_model(base_path / "faceFeatureModel.h5")


def getLipFeatureModel():
    '''
    returns the pretrained Lip Feature Model
    '''
    return load_model(base_path / "lipFeatureModel.h5")


def getFaceImageModelPath():
    '''
    returns the path to pretrained Face Image Model
    '''
    return base_path / "bestFaceEndToEnd.h5"


def getLipImageModelPath():
    '''
    returns the path to pretrained Lip Image Model
    '''
    return base_path / "bestLipEndToEnd.h5"


def getFaceFeatureModelPath():
    '''
    returns the path to pretrained Face Feature Model
    '''
    return base_path / "faceFeatureModel.h5"


def getLipFeatureModelPath():
    '''
    returns the path to pretrained Lip Feature Model
    '''
    return base_path / "lipFeatureModel.h5"
