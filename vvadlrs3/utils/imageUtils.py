"""utils for images"""

# System imports
import os
import pathlib
import argparse
from multiprocessing import Process
import shutil

# 3rd party imports
import cv2
import dlib
import numpy as np

# local imports

def getRandomFaceFromImage(image):
    """
    returns the bounding box for first face from an image
    """
    detector = dlib.get_frontal_face_detector()
    dets = detector(image, 1)
    if dets:
        return dets[0]
    else:
        return False

def cropImage(image, drect):
    """
    crop the image with a drect
    """
    # make rect from drect
    rect = dlib.rectangle(drect)
    #print("ROI shape: {}\nlipRect: {}".format(image.shape, rect))
    # crop with max(width, heigh) and min(0)
    maxX = image.shape[1] - 1
    maxY = image.shape[0] - 1
    return image[np.clip(rect.top(), 0, maxY):np.clip(rect.bottom(), 0, maxY), np.clip(rect.left(), 0, maxX):np.clip(rect.right(), 0, maxX)]
def toImageSpace(ROIrect, rect):
    """
    returns a rectangle in the coordinate space of the whole image

    :param ROIrect: rectangle in Image Space, which contains the rect
    :type ROIrect: dlib.drectangle
    :param rect: rectangle in ROI Space
    :type ROI: dlib.drectangle
    """
    return dlib.drectangle(ROIrect.left() + rect.left(), ROIrect.top() + rect.top(), ROIrect.left() + rect.right(), ROIrect.top() + rect.bottom())

def resizeAndZeroPadding(image, shape):
    """
    Resizes a given image to the disered shape without changing the (width to height)ratio.
    (On pixel base its not suitable to resize to the exact same ratio - closest with rounding is applied)
    Applies zeroPadding to ensure the desired size.
    :param image: The image for resizing
    :type image: numpyarray from openCV
    :param shape: the desired shape(width, height)
    :type shape: tuple of ints
    :returns: resized image
    """
    (h, w) = image.shape[:2]
    widthFaktor = shape[0] / w
    heightFaktor = shape[1] / h
    faktor = min([widthFaktor, heightFaktor])
    #print("Faktor: {}".format(faktor))
    multipliedSize = (int(round(w * faktor)), int(round(h * faktor)))
    #print ("Multpilied size: {}".format(multipliedSize))

    image = cv2.resize(image, multipliedSize)
    #cant be exactly the same ratio because resize can only resize to full pixels - take big images(LAW of big numbers will take care)
    #print("Ratio disortion: {}".format(abs(float(image.shape[1]) / image.shape[0] - float(w) / h)))
    #print("new shape: {}".format(image.shape))
    widthDelta = abs(image.shape[1] - shape[0])
    heightDelta = abs(image.shape[0] - shape[1])
    image = cv2.copyMakeBorder(image,0,heightDelta,0,widthDelta,cv2.BORDER_CONSTANT,value=[0,0,0])
    #image = np.pad(image, [(0, heightDelta), (0, widthDelta)], 'constant')
    assert min([heightDelta, widthDelta]) == 0, "Padding must only be aplied on one side. widthDelta: {}, heightDelta: {}".format(widthDelta, heightDelta)
    assert image.shape[0] == shape[1], "Desired height was {} transformed to {}".format(shape[1], image.shape[0])
    assert image.shape[1] == shape[0], "Desired width was {} transformed to {}".format(shape[0], image.shape[1])

    return image

def convertSampleToVideo(data, path, fps=25, codec = 'MP4V'):
    """
    converts a sample(data) to a video stream.

    :param data: numpy array of shape (timesteps, *img.shape)
    :type data: numpy array
    :param path: path to write the video to 
    :type path: String
    """
    size = (data.shape[1], data.shape[2])
    out = cv2.VideoWriter()
    out.open(path, cv2.VideoWriter_fourcc(*codec), fps, size, True)
    for img in data:
        # convert to BGR
        # img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR) - seems to be not the Problem here
        out.write(img)
    out.release()
