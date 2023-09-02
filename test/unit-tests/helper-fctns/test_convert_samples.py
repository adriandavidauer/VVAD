import os
import unittest
import shutil

from vvadlrs3.convert_samples import convert_samples as sample_conv


class TestSampleConverter(unittest.TestCase):
    """
        Test the conversion of samples to jpg data
    """

    def setUp(self):
        self.test_data_root = "test/unit-tests/helper-fctns/testData"

    def test_convert_samples(self):
        input_path = self.test_data_root
        output_path = self.test_data_root + "generatedImages"
        number_samples = 5
        sample_conv(input_path, output_path, number_samples)

        # evaluate number of generated images in folder
        self.assertEqual(number_samples, len(os.listdir(output_path)),
                         "Amount of generated images does "
                         "not match expected number!")

        self.assertTrue(os.path.exists(output_path))
        shutil.rmtree(output_path)


if __name__ == '__main__':
    unittest.main()
