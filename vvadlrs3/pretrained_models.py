'''
This Module has all the pretrained Models for the VVAD-LRS3
'''
# System imports

# 3rd party imports
from keras.models import load_model


# local imports

__author__      = 'Adrian Lubitz'

def getFaceImageModel():
    '''
    returns the pretrained Face Image Model
    '''
    return  load_model("bestFaceEndToEnd.h5")

def getLipImageModel():
    '''
    returns the pretrained Lip Image Model
    '''
    return  load_model("bestLipEndToEnd.h5")

def getFaceFeatureModel():
    '''
    returns the pretrained Face Feature Model
    '''
    return  load_model("faceFeatureModel.h5")

def getLipFeatureModel():
    '''
    returns the pretrained Lip Feature Model
    '''
    return  load_model("lipFeatureModel.h5")