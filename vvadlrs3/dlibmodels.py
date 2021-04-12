'''This module offers scripts to download bigger models on the fly from the official dlib-models repo on github(https://github.com/davisking/dlib-models)'''
# System imports
import os
from pathlib import Path
import urllib.request
import bz2
import errno
import os


# 3rd Party imports
from tqdm import tqdm

# local imports
from vvadlrs3.utils.downloadUtils import download_url

# end file header

__author__ = 'Adrian Lubitz'


def SHAPE_PREDICTOR_68_FACE_LANDMARKS():
    predictor_path = Path(__file__).absolute().parent / \
        'shape_predictor_68_face_landmarks.dat'
    compressed_file = Path(predictor_path.parent /
                           (predictor_path.name + '.bz2'))
    if not predictor_path.exists():
        download_url('https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2',
                     compressed_file)

        with open(predictor_path, 'wb') as new_file, bz2.BZ2File(compressed_file, 'rb') as file:
            for data in iter(lambda: file.read(100 * 1024), b''):
                new_file.write(data)

        # remove compressed file
        compressed_file.unlink()

        if predictor_path.exists():
            return predictor_path
        else:  # This case should never happen! is only possible if file is deleted externally
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), predictor_path)
    else:
        return predictor_path
