import unittest
import sys
import tempfile
from pathlib import Path
import numpy as np
from vvadlrs3 import sample as sample



class TestFaceTracker(unittest.TestCase):
    def test_get_next_face(self):
        pass


class TestFaceFeatureGenerator(unittest.TestCase):
    def test_get_features(self):
        pass

    def test_is_valid(self):
        # Use arbitrary np array to do proof of concept
        test_sample = sample.FeatureizedSample()
        test_sample.data = [np.ones((2, 2))] * 5
        test_sample.k = 5
        self.assertTrue(test_sample.isValid())
        test_sample.k = 2
        self.assertFalse(test_sample.isValid())

    def test_get_data(self):
        pass

    def test_get_dist(self):
        pass

    def test_normalize(self):
        pass

    def test_get_label(self):
        pass

    def test_generate_sample_from_fixed_frames(self):
        pass

    def test_generate_sample_from_buffer(self):
        pass

    def test_visualize(self):
        pass

    def test_save_data_as_pickle(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            temp_dir = Path(tmpdirname)
            data_dir = temp_dir.joinpath(self, "test.pickle")
            sample.FeatureizedSample.save(data_dir)

            self.assertEqual(data_dir.exists(), True)
            # ToDo check if file is empty

    def test_load_existing_pickle(self):
        # ToDo: Check exception type
        sample_path = "./testData/testSample.pickle"
        try:
            sample.FeatureizedSample.load(sample_path)
        except ValueError:
            self.fail("ValueError raised unexpectedly!")

    def test_load_non_existing_pickle(self):
        # ToDo: Check exception type
        sample_path = "./testData/noTestSample.pickle"
        self.assertRaises(ValueError, sample.FeatureizedSample.load(sample_path))


class TestFeaturedSample(unittest.TestCase):
    pass


class TestVisualizeSamples(unittest.TestCase):
    def test_visualize_samples(self):
        sample.visualizeSamples()
