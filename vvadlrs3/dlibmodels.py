"""
    This module offers scripts to download bigger models on the fly from the
    official dlib-models repo on github(https://github.com/davisking/dlib-models)
"""
import bz2
import errno
import os
# System imports
from pathlib import Path

# local imports
from vvadlrs3.utils.downloadUtils import download_url

# 3rd Party imports

# end file header

__author__ = 'Adrian Lubitz'


def SHAPE_PREDICTOR_68_FACE_LANDMARKS():
    """

    This predictor is part of the dlib-models repository
    (https://github.com/davisking/dlib-models) and was trained on the ibug 300-W
    dataset.

    This  model file is designed for use with dlib's HOG face detector. That is, it
    expects the bounding boxes from the face detector to be aligned a certain way,
    the way dlib's HOG face detector does it. It won't work as well when used with a
    face detector that produces differently aligned boxes.

    Returns:
        predictor_path (str): Path to the predictor

    """
    predictor_path = Path(__file__).absolute().parent / \
        'shape_predictor_68_face_landmarks.dat'
    compressed_file = Path(predictor_path.parent /
                           (predictor_path.name + '.bz2'))
    if not predictor_path.exists():
        download_url('https://github.com/davisking/dlib-models/raw/master/'
                     'shape_predictor_68_face_landmarks.dat.bz2',
                     compressed_file)

        with open(predictor_path, 'wb') as new_file, bz2.BZ2File(compressed_file,
                                                                 'rb') as file:
            for data in iter(lambda: file.read(100 * 1024), b''):
                new_file.write(data)

        # remove compressed file
        compressed_file.unlink()

        if predictor_path.exists():
            return predictor_path
        else:  # This case should never happen! is only possible if file is deleted
            # externally
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), predictor_path)
    else:
        return predictor_path
