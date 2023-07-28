"""
This is an experiment to analyze videos with different fps and feature_types
"""

# System imports
import argparse
import json
import glob
from pathlib import Path
import sys


# 3rd party imports
from vvadlrs3.utils import videoUtils, plotUtils


# local imports

# end file header
__author__ = 'Adrian Lubitz'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)

    action.add_argument(
        "--video_path", help='Path to the video. If this is given from_json will be '
                             'ignored. If a folder is given all '
                             'videos will be analyzed and saved to json files with '
                             'the same name.', type=str)
    action.add_argument(
        "--from_json", help="Path to the json file where you want to load the "
                            "analysis from. If a folder is given all "
                            "json files in it will be analyzed and images with the "
                            "same name will be saved.")
    parser.add_argument("-f", "--feature_type", help="type of the features that "
                                                     "should be used when creating "
                                                     "samples.",
                        choices=["faceImage", "lipImage", "faceFeatures",
                                 "lipFeatures"], type=str, default='faceImage')
    parser.add_argument(
        "--save_json", help="Path where to save the analysis as json file", type=str)
    # TODO normally you would always want that or at least one of --save_json or
    #  visualize
    parser.add_argument(
        "-v", "--visualize", help="Visualize the analysis", action='store_true')

    args = parser.parse_args()

    if args.video_path:
        video_path = Path(args.video_path)
        if video_path.is_file():
            analysis = videoUtils.analyze_video(
                args.video_path, args.feature_type, save_as_json=args.save_json)
        elif video_path.is_dir():
            # get all video files
            types = ('*.mkv', '*.mp4', '*.webm')  # the tuple of file types
            videos = []
            for files in types:
                video_path = Path(args.videos_from_folder) / files
                videos.extend(
                    glob.glob(str(video_path)))
            # loop over all video files and analyze
            for video in videos:
                save_as_json = Path(video).parent / \
                    (Path(video).name + f'.{args.feature_type}.json')
                videoUtils.analyze_video(
                    video, args.feature_type, save_as_json=save_as_json)

            # stop execution because visualization is not possible in this case
            sys.exit()
    elif args.from_json:
        from_json = Path(args.from_json)
        if from_json.is_file():
            with open(args.from_json) as json_file:
                analysis = json.load(json_file)
        elif from_json.is_dir():
            all_json_files = list(glob.glob(str(from_json / '*.json')))
            for analysis_file in all_json_files:
                print(analysis_file)
                with open(analysis_file) as json_file:
                    analysis = json.load(json_file)
                    # create visualization for all of them and save the fig under same
                    # name.png
                    plotUtils.plot_video_analysis(
                        analysis, show=False, path=analysis_file + '.png')

    if args.visualize:
        plotUtils.plot_video_analysis(analysis)
