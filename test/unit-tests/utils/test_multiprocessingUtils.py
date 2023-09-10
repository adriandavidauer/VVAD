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

        self.positives_folder = os.path.join(self.test_data_root,
                                             "data/positiveSamples")
        self.negatives_folder = os.path.join(self.test_data_root,
                                             "data/negativeSamples")
        if not os.path.exists(self.positives_folder):
            os.makedirs(self.positives_folder)
        if not os.path.exists(self.negatives_folder):
            os.makedirs(self.negatives_folder)

    """"""

    def test_producer_consumer(self):
        current_folder = self.test_data_root + "/data/0af00UcTOSc"

        feature_type = "faceImage"
        data_shape = [200, 200]
        multiprocessingUtils.producer(self.data_set,
                                      [current_folder, feature_type,
                                       data_shape])

        self.ratio_positives = multiprocessingUtils.positivesQueue.qsize()
        self.ratio_negatives = multiprocessingUtils.negativesQueue.qsize()

        multiprocessingUtils.consumer(self.positives_folder, self.negatives_folder,
                                      self.ratio_positives, self.ratio_negatives)

        # evaluate positive sample pickles
        for file in range(len(os.listdir(self.positives_folder))):
            self.assertTrue(
                os.path.exists(os.path.join(
                    self.positives_folder, str(file) + ".pickle"
                ))

            )

        # evaluate negative sample pickles
        for file in range(len(os.listdir(self.negatives_folder))):
            self.assertTrue(
                os.path.exists(os.path.join(
                    self.negatives_folder, str(file) + ".pickle"
                ))

            )

        shutil.rmtree(self.positives_folder)
        shutil.rmtree(self.negatives_folder)


if __name__ == '__main__':
    unittest.main(warnings='ignore')