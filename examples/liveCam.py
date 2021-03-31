'''
This is a live Demo using the webcam
'''

# System imports

# 3rd party imports
from vvadlrs3 import pretrained_models
from vvadlrs3 import sample
import cv2


# local imports

# end file header
__author__ = 'Adrian Lubitz'

# Sliding window approach

# load pretrained model
model = pretrained_models.getFaceImageModel()


# create a webcambuffer
cap = cv2.VideoCapture(0)


# create a sample
# create_sample_from_buffer
# infer if sample is valid
# annotate and visualize output image(first of sample data) - do I need to save the boundingbox as well in the sample?

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
