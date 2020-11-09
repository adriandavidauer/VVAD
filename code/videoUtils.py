"""utils for videos"""

# System imports
import os
import pathlib
import argparse
from multiprocessing import Process
import shutil

# 3rd party imports
from pytube import YouTube
import cv2
from ffmpy import FFmpeg
import dlib
import numpy as np
from file_read_backwards import FileReadBackwards
import matplotlib.pyplot as plt
import yaml

# local imports

def getFramesfromVideo(videoPath):
    """
    yields the frames from a video
    """
    success = True
    vidObj = cv2.VideoCapture(str(videoPath))
    while(success):
        success, image = vidObj.read()
        yield success, image
