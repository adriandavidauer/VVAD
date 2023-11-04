# import the necessary packages
from collections import OrderedDict
import numpy as np
import cv2

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions

# For dlib’s 68-point facial landmark detector:
FACIAL_LANDMARKS_68 = OrderedDict([
    ("mouth", (48, 68)),
    ("inner_mouth", (60, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])

# in order to support legacy code, we'll default the indexes to the
# 68-point model
FACIAL_LANDMARKS = FACIAL_LANDMARKS_68


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y, z)-coordinates
    coords = np.zeros((len(shape), 3), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 3-tuple of (x, y, z)-coordinates
    for i in range(0, len(shape)):
        coords[i] = (shape[i][0], shape[i][0], shape[i][0])

    # return the list of (x, y, z)-coordinates
    return coords
