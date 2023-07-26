import os.path
import unittest
import sys
sys.path.append('../../../vvadlrs3')
from vvadlrs3 import dataSet as dSet

"""
    All tests are run from the unit-tests folder as root! This must be considered when running these tests 
    with the GitHub action runner
"""


class TestDataSet(unittest.TestCase):

    def setUp(self):
        self.test_data_root = "test/unit-tests/data-and-models/testData"
        self.video_folder_path = "video/00j9bKdiOjk"
        self.video_file_path = "video/00j9bKdiOjk/00j9bKdiOjk.3gpp"
        self.data_set = dSet.DataSet(shapeModelPath="",
                                     debug=True,
                                     sampleLength=0,
                                     maxPauseLength=1.5,
                                     shape=None,
                                     path=None,
                                     fps=25,
                                     multiprocessing=False
                                     )

    def test_download_LRS3_sample_from_yt(self):
        self.data_set.downloadLRS3SampleFromYoutube(path=os.path.join(self.test_data_root, self.video_folder_path))
        self.assertTrue(os.path.exists(os.path.join(self.test_data_root, self.video_file_path)))
        os.remove(os.path.join(self.test_data_root, self.video_file_path))

    @unittest.expectedFailure
    def test_download_LRS3_from_yt_wrong_path(self):
        self.assertRaises(dSet.WrongPathException, callable=self.data_set.downloadLRS3SampleFromYoutube(
            path=os.path.join(self.test_data_root, "video/noVideoFolder")))

    def test_get_all_positive_samples(self):
        pass

    def test_get_all_samples(self):
        pass

    def test_convert_all_fps(self):
        pass

    def test_download_lrs3(self):
        pass

    def test_get_txt_files(self):
        for textfile in self.data_set.getTXTFiles(path=os.path.join(self.test_data_root, "getTXT")):
            self.assertTrue(str(textfile).__contains__("myTXT.txt"))

    def test_fail_get_txt_files(self):
        self.assertRaises(dSet.WrongPathException,
                          callable=self.data_set.getTXTFiles(path=os.path.join(self.test_data_root, "getNoTXTs")))

    # ToDo: check if same as test_get_all_positive_samples
    #@unittest.expectedFailure
    def test_get_positive_samples(self):
        # ToDo somehow wrong

        # data_set.downloadLRS3SampleFromYoutube(path=folder_path)
        self.data_set.getPositiveSamples(path=os.path.join(self.test_data_root, self.video_folder_path), dryRun=False)
        print(self.data_set.getPositiveSamples(path=os.path.join(self.test_data_root, self.video_folder_path),
                                               dryRun=True))
        for sample in dSet.DataSet.getPositiveSamples(os.path.join(self.test_data_root, self.video_folder_path), True):
            print("hey")
            yield sample
        print("[getAllPSamples] Folder {} done".format(os.path.join(self.test_data_root, self.video_folder_path)))

    # ToDo: check if same as test_convert_all_fps
    def test_convert_fps(self):
        pass

    def test_get_video_path_from_video(self):
        pass

    def test_analyze_negatives(self):
        pass

    def test_analyze_positives(self):
        pass

    def test_get_frame_from_second(self):
        self.assertEqual(self.data_set.getFrameFromSecond(second=15.5, fps=25), 387.5)

    def test_get_second_from_frame(self):
        self.assertEqual(self.data_set.getSecondFromFrame(frame=1500, fps=25), 60.0)

    def test_get_pause_length(self):
        self.data_set.maxPauseLength = 0.5
        self.data_set.sampleLength = 25
        print(self.data_set.getPauseLength(txtFile=os.path.join(self.test_data_root, "pause_example.txt")))
        self.assertEqual(self.data_set.getPauseLength(txtFile=os.path.join(self.test_data_root, "pause_example.txt")),
                         [(26.14, 27.64), (17.04, 18.01), (11.29, 12.0), (3.55, 5.63)])

    def test_get_no_pause_length(self):
        self.assertEqual(
            self.data_set.getPauseLength(txtFile=os.path.join(self.test_data_root, "no_pause_example.txt")), [])

    def test_get_sample_configs_for_positive_samples(self):
        pass

    def test_check_sample_length(self):
        pass

    def test_get_sample_configs(self):
        pass

    def test_get_samples(self):
        pass

    def test_analyze(self):
        pass

    def test_grap_from_video(self):
        pass

    def test_grap_from_disk(self):
        pass


class TestSaveBalancedDataset(unittest.TestCase):
    def test_save_balanced_dataset(self):
        pass


class TestTransformations(unittest.TestCase):
    def test_transform_to_hdf5(self):
        pass

    def test_transform_points_to_numpy(self):
        pass

    def test_transform_to_features(self):
        pass


class TestMakeTestSet(unittest.TestCase):
    def test_make_test_set(self):
        pass


if __name__ == "__main__":
    print("name", __name__)
    unittest.main()
