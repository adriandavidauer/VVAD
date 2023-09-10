import os.path
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from vvadlrs3 import pretrained_models


def get_rgb_test_image(image_file_name, folder_path):
    img = cv2.imread(os.path.join(folder_path, image_file_name))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class TestPretrainedModelPath(unittest.TestCase):

    def test_get_face_img_model_path(self):
        self.assertEqual(
            str(Path(__file__).absolute().parent) + "/bestFaceEndToEnd.h5",
            pretrained_models.get_face_img_model_path()
        )

    def test_get_lip_img_model_path(self):
        self.assertEqual(
            str(Path(__file__).absolute().parent) + "/bestLipEndToEnd.h5",
            pretrained_models.get_lip_img_model_path()
        )

    def test_get_face_feature_model_path(self):
        self.assertEqual(
            str(Path(__file__).absolute().parent) + "/faceFeatureModel.h5",
            pretrained_models.get_face_feature_model_path()
        )

    def test_get_lip_feature_model_path(self):
        self.assertEqual(
            str(Path(__file__).absolute().parent) + "/lipFeatureModel.h5",
            pretrained_models.get_lip_feature_model_path()
        )


if __name__ == '__main__':
    unittest.main(warnings='ignore')
