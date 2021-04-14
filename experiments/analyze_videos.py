'''
This is an experiment to analyze videos with different fps and feature_types
'''

# System imports
import argparse


# 3rd party imports
from vvadlrs3.utils import videoUtils


# local imports

# end file header
__author__ = 'Adrian Lubitz'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help='Path to the video', type=str)
    parser.add_argument("-f", "--feature_type", help="type of the features that should be used when creating samples.",
                        choices=["faceImage", "lipImage", "faceFeatures", "lipFeatures"], type=str, default='faceImage')
    parser.add_argument(
        "--save_json", help="Path where to save the analysis as json file", type=str)
    parser.add_argument(
        "-v", "--visualize", help="Visualize the analysis", action='store_true')  # TODO normally you would always want that or at least one of --save_json or visualize

    args = parser.parse_args()

    videoUtils.analyzeVideo(
        args.video_path, args.feature_type, save_as_json=args.save_json)
