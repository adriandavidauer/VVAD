import os
import shutil
import sys
import unittest

sys.path.append('../../../vvadlrs3')
from vvadlrs3 import dataSet as dSet
from vvadlrs3.utils import multiprocessingUtils


class TestMultiprocessingUtils(unittest.TestCase):

    def setUp(self) -> None:
        self.test_data_root = "test/unit-tests/utils/testData"
        self.data_set = dSet.DataSet(
            debug_flag=False,
            sample_length=1.0,
            max_pause_length=2.0,
            init_shape=[200, 200],
            path=None,
            target_fps=25,
            init_multiprocessing=True,
            shape_model_path="../../../models"
                             "/shape_predictor_5_face_landmarks.dat "
        )

        print("Set up folders")
        self.positives_folder = os.path.join(self.test_data_root,
                                             "data/positiveSamples")
        self.negatives_folder = os.path.join(self.test_data_root,
                                             "data/negativeSamples")
        if not os.path.exists(self.positives_folder):
            os.makedirs(self.positives_folder)
        if not os.path.exists(self.negatives_folder):
            os.makedirs(self.negatives_folder)
        self.ratio_positives = 2
        self.ratio_negatives = 0

    """"""

    def test_producer_consumer(self):
        current_folder = self.test_data_root + "/data/0af00UcTOSc"

        feature_type = "faceImage"
        data_shape = [200, 200]
        multiprocessingUtils.producer(self.data_set,
                                      [current_folder, feature_type,
                                       data_shape])

        multiprocessingUtils.consumer(self.positives_folder, self.negatives_folder,
                                      self.ratio_positives, self.ratio_negatives)

        # No suitable negative samples in this video detected
        # 2 positive samples expected
        self.assertTrue(
            os.path.exists(os.path.join(self.positives_folder, "0.pickle")))
        self.assertTrue(
            os.path.exists(os.path.join(self.positives_folder, "1.pickle")))

        shutil.rmtree(self.positives_folder)
        shutil.rmtree(self.negatives_folder)
        os.remove(os.path.join(current_folder, "0af00UcTO-c.converted.3gp"))
