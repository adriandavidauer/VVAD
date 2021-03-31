'''
This is a live Demo using the webcam
'''

# System imports

# 3rd party imports
from vvadlrs3 import pretrained_models

# local imports

# end file header
__author__ = 'Adrian Lubitz'

# Sliding window approach

# load pretrained model
model = pretrained_models.getFaceImageModel()

# create a sample
# create a webcambuffer
# create_sample_from_buffer
# infer if sample is valid
# annotate and visualize output image(first of sample data) - do I need to save the boundingbox as well in the sample?
