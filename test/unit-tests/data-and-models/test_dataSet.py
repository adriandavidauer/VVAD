import os.path
import pathlib
import shutil
import sys
import unittest

import numpy

sys.path.append('../../../vvadlrs3')
from vvadlrs3 import dataSet as dSet

"""
    All tests are run from the unit-tests folder as root! This must be considered when 
    running these tests 
    with the GitHub action runner
"""


class TestDataSet(unittest.TestCase):

    def setUp(self):
        self.test_data_root = "test/unit-tests/data-and-models/testData"
        self.videos_path = "video"
        self.video_folder_path = "video/00j9bKdiOjk"
        self.video_folder_path_2 = "video/0Amg53UuRqE"
        self.video_folder_path_3 = "video/0akiEFwtkyA"
        self.video_file_path = "video/00j9bKdiOjk/00j9bKdiOjk.mp4"
        self.video_file_path_converted = "video/00j9bKdiOjk/00j9bKdiOjk.coverted.mp4"
        self.video_txt_file_path = "video/00j9bKdiOjk/00001.txt"
        self.data_set = dSet.DataSet(
            debug_flag=True,
            sample_length=2.0,
            max_pause_length=1.5,
            init_shape=None,
            path=None,
            target_fps=25,
            init_multiprocessing=False,
            shape_model_path="../../../models/shape_predictor_5_face_landmarks.dat "
        )

    def test_download_LRS3_sample_from_yt(self):
        self.data_set.download_lrs3_sample_from_youtube(path=os.path.join(
            self.test_data_root, self.video_folder_path))
        self.assertTrue(os.path.exists(os.path.join(self.test_data_root,
                                                    self.video_file_path)))
        os.remove(os.path.join(self.test_data_root, self.video_file_path))

    def test_download_LRS3_from_yt_wrong_path(self):
        self.assertRaises(dSet.WrongPathException,
                          lambda: self.data_set.download_lrs3_sample_from_youtube(
                              path=os.path.join(self.test_data_root,
                                                "video/noVideoFolder")))

    @unittest.expectedFailure
    def test_get_all_positive_samples(self):
        # Preparation
        self.data_set.download_lrs3_sample_from_youtube(path=os.path.join(
            self.test_data_root, self.video_folder_path))
        self.data_set.download_lrs3_sample_from_youtube(path=os.path.join(
            self.test_data_root, self.video_folder_path_2))

        path = os.path.join(self.test_data_root, self.videos_path)

        # Execution and first evaluation
        for sample in self.data_set.get_all_p_samples(path):
            self.assertIsNotNone(sample)

        video_path_1 = "video/0Amg53UuRqE/0Amg53UuRqE.mp4"
        video_path_2 = "video/00j9bKdiOjk/00j9bKdiOjk.mp4"

        # Evaluation
        self.assertTrue(os.path.exists(os.path.join(self.test_data_root,
                                                    video_path_1)))
        self.assertTrue(os.path.exists(os.path.join(self.test_data_root,
                                                    video_path_2)))

        # Clean
        folders = list(os.walk(path, followlinks=True))[0][1]
        folders.sort()
        for folder in folders:
            files = os.listdir(os.path.join(self.test_data_root, self.videos_path,
                                            folder))
            for item in files:
                if item.endswith(".mp4") or item.endswith(".3gpp"):
                    os.remove(os.path.join(self.test_data_root, self.videos_path,
                                           folder, item))

    def test_get_all_samples(self):
        self.data_set.get_all_samples(
            path=os.path.join(self.test_data_root, "video").replace("\\", "/"),
            showStatus=True,
            feature_type=None)

    def test_convert_all_fps(self):
        # Windows needs ffmpeg.exe as executable. Might not be needed for Linux
        self.data_set.download_lrs3(path=os.path.join(
            self.test_data_root, self.videos_path))
        self.data_set.convert_all_fps(
            os.path.join(self.test_data_root, self.videos_path).replace("\\", "/"))

        self.assertEqual(25, self.data_set.fps)

        # Clean
        path = os.path.join(self.test_data_root, self.videos_path)

        folders = list(os.walk(path, followlinks=True))[0][1]
        folders.sort()
        for folder in folders:
            files = os.listdir(os.path.join(self.test_data_root, self.videos_path,
                                            folder))
            for item in files:
                if item.endswith(".mp4") or item.endswith(".3gpp"):
                    os.remove(os.path.join(self.test_data_root, self.videos_path,
                                           folder, item))

    def test_download_lrs3(self):
        self.data_set.download_lrs3(path=os.path.join(
            self.test_data_root, self.videos_path))

        self.assertTrue(
            os.path.exists(
                os.path.join(self.test_data_root, self.video_folder_path,
                             "00j9bKdiOjk.mp4")
            )
        )

        self.assertTrue(
            os.path.exists(
                os.path.join(self.test_data_root, self.video_folder_path_2,
                             "0Amg53UuRqE.mp4")
            )
        )

        # Clean
        path = os.path.join(self.test_data_root, self.videos_path)

        folders = list(os.walk(path, followlinks=True))[0][1]
        folders.sort()
        for folder in folders:
            files = os.listdir(os.path.join(self.test_data_root, self.videos_path,
                                            folder))
            for item in files:
                if item.endswith(".mp4") or item.endswith(".3gpp"):
                    os.remove(os.path.join(self.test_data_root, self.videos_path,
                                           folder, item))

    def test_get_txt_files(self):
        for textfile in self.data_set.get_txt_files(
                 os.path.join(self.test_data_root, "getTXT")):
            self.assertTrue(str(textfile).__contains__("myTXT.txt"))

    def test_get_positive_samples_dry(self):
        for sample in self.data_set.get_positive_samples(os.path.join(
                self.test_data_root, self.video_folder_path),
                True):
            self.assertIsNotNone(sample)
        print("[getAllPSamples] Folder {} done".format(os.path.join(
            self.test_data_root, self.video_folder_path)))

    @unittest.expectedFailure
    def test_get_positive_samples(self):
        self.data_set.download_lrs3_sample_from_youtube(path=os.path.join(
            self.test_data_root, self.video_folder_path))
        self.assertTrue(os.path.exists(os.path.join(self.test_data_root,
                                                    self.video_file_path)))

        for sample in self.data_set.get_positive_samples(os.path.join(
                self.test_data_root, self.video_folder_path),
                False):
            print(sample)
        print("[getAllPSamples] Folder {} done".format(os.path.join(
            self.test_data_root, self.video_folder_path)))

        # Clean
        os.remove(os.path.join(self.test_data_root, self.video_file_path))
        files = os.listdir(os.path.join(self.test_data_root, self.video_folder_path))
        for item in files:
            if item.endswith(".3gpp") or item.endswith(".mp4"):
                os.remove(os.path.join(self.test_data_root, self.video_folder_path,
                                       item))

    def test_get_no_video_path_from_folder_index_error(self):
        video_path = os.path.join(self.test_data_root, "noVideoData"). \
            replace("\\", "/")
        self.assertRaises(dSet.WrongPathException,
                          lambda: self.data_set.get_video_path_from_folder(video_path))

    def test_get_no_video_path_from_folder(self):
        video_path = os.path.join(self.test_data_root, "getTXT"). \
            replace("\\", "/")
        self.assertRaises(FileNotFoundError,
                          lambda: self.data_set.get_video_path_from_folder(video_path))

    def test_get_video_path_from_folder(self):
        video_path = os.path.join(self.test_data_root, "test_video_data"). \
            replace("\\", "/")

        self.assertEqual(self.data_set.get_video_path_from_folder(video_path).__str__(),
                         os.path.join(self.test_data_root,
                                      pathlib.Path(
                                          os.path.join(os.path.abspath(video_path),
                                                       "test_video.mp4"))))

    def test_too_many_videos_in_folder(self):
        src_dir = os.path.join(self.test_data_root, "test_video_data",
                               "test_video.mp4"). \
            replace("\\", "/")
        dst_dir = os.path.join(self.test_data_root, "test_video_data",
                               "test_video.converted.mp4"). \
            replace("\\", "/")
        shutil.copy(src_dir, dst_dir)

        self.assertTrue(os.path.exists(dst_dir))

        video_path = os.path.join(self.test_data_root, "test_video_data"). \
            replace("\\", "/")

        self.data_set.get_video_path_from_folder(pathlib.Path(video_path))

        self.assertFalse(os.path.exists(dst_dir))

    def test_analyze_negatives(self):
        pauses = self.data_set.analyze_negatives(
            path=os.path.join(self.test_data_root, self.videos_path),
            save_to=os.path.join(self.test_data_root, "analyze_negatives"))

        self.assertEqual(len(pauses), 4)
        self.assertTrue(os.path.exists(
            os.path.join(self.test_data_root,
                         "analyze_negatives.png")))
        os.remove(os.path.join(self.test_data_root,
                               "analyze_negatives.png"))

    def test_analyze_positives(self):
        self.data_set.download_lrs3_sample_from_youtube(path=os.path.join(
            self.test_data_root, self.video_folder_path))

        positives, num_samples = self.data_set.analyze_positives(
            path=os.path.join(self.test_data_root, self.videos_path),
            save_to=os.path.join(self.test_data_root, "analyze_positives"))

        self.assertEqual(num_samples, 3)
        self.assertTrue(os.path.exists(
            os.path.join(self.test_data_root,
                         "analyze_positives.png")))
        os.remove(os.path.join(self.test_data_root,
                               "analyze_positives.png"))
        # Clean
        path = os.path.join(self.test_data_root, self.videos_path)

        folders = list(os.walk(path, followlinks=True))[0][1]
        folders.sort()
        for folder in folders:
            files = os.listdir(os.path.join(self.test_data_root, self.videos_path,
                                            folder))
            for item in files:
                if item.endswith(".mp4") or item.endswith(".3gpp"):
                    os.remove(os.path.join(self.test_data_root, self.videos_path,
                                           folder, item))

    def test_get_frame_from_second(self):
        # default fps = 25
        self.assertEqual(self.data_set.get_frame_from_second(second=15.5), 387.5)

    def test_get_second_from_frame(self):
        # default fps = 25
        self.assertEqual(self.data_set.get_second_from_frame(frame=1500), 60.0)

    def test_get_pause_length(self):
        self.data_set.maxPauseLength = 0.5
        self.data_set.sampleLength = 25
        print(self.data_set.get_pause_length(
            txt_file=os.path.join(self.test_data_root, "pause_example.txt")))
        self.assertEqual(self.data_set.get_pause_length(
            txt_file=os.path.join(self.test_data_root, "pause_example.txt")),
            [(26.14, 27.64), (17.04, 18.01), (11.29, 12.0), (3.55, 5.63)])

    def test_get_no_pause_length(self):
        self.assertEqual(
            self.data_set.get_pause_length(
                txt_file=os.path.join(self.test_data_root, "no_pause_example.txt")), [])

    def test_get_sample_configs_for_positive_samples(self):
        config = self.data_set.get_sample_configs_for_pos_samples(
            txt_file=os.path.join(self.test_data_root, "pause_example.txt")
        )

        # [startFrame, endFrame , x, y, w, h] x,y,w,h are relative pixels
        self.assertEqual(config[0][0], 2120, "startFrame matches")
        self.assertEqual(config[0][1], 2209, "endFrame matches")
        self.assertEqual(config[0][2], .456,
                         "relative pixels in x according to expectation")
        self.assertEqual(config[0][3], .108,
                         "relative pixels in y according to expectation")
        self.assertEqual(config[0][4], .103,
                         "relative pixels in w according to expectation")
        self.assertEqual(config[0][5], .271,
                         "relative pixels in h according to expectation")

    def test_check_sample_length(self):
        self.assertTrue(self.data_set.check_sample_length(3.55, 5.63))

    def test_get_sample_configs(self):
        # [(label, [startFrame, endFrame , x, y, w, h]), ...] x,y,w,h are relative
        # pixels of the bounding box in thh first frame

        self.assertEqual(self.data_set.get_sample_configs(txt_file=os.path.join(
            self.test_data_root, self.video_txt_file_path)),
            [(True, [932, 981, 0.382, 0.138, 0.105, 0.277]),
             (True, [982, 1031, 0.445, 0.14, 0.105, 0.267]),
             (True, [1032, 1081, 0.437, 0.137, 0.106, 0.268]),
             (True, [1082, 1131, 0.407, 0.135, 0.104, 0.28])])

    # ToDo Test not finished
    def test_get_samples(self):
        samples = self.data_set.get_samples(
            path=os.path.join(self.test_data_root, self.video_folder_path),
            feature_type="faceImage",
            samples_shape=(200, 200))

        print(samples)
        # self.data_set.get_samples(path=os.path.join(self.test_data_root,
        # self.video_folder_path).replace("\\", "/"),
        #                          feature_type=,
        #                          samples_shape=)

    @unittest.expectedFailure
    def _test_analyze(self):
        # Windows needs ffmpeg.exe as executable. Might not be needed for Linux
        self.data_set.download_lrs3_sample_from_youtube(path=os.path.join(
            self.test_data_root, self.video_folder_path))
        self.data_set.download_lrs3_sample_from_youtube(path=os.path.join(
            self.test_data_root, self.video_folder_path_2))
        self.data_set.download_lrs3_sample_from_youtube(path=os.path.join(
            self.test_data_root, self.video_folder_path_3))

        self.data_set.analyze(path=os.path.join(self.test_data_root, self.videos_path))

        # Clean
        path = os.path.join(self.test_data_root, self.videos_path)

        folders = list(os.walk(path, followlinks=True))[0][1]
        folders.sort()
        for folder in folders:
            files = os.listdir(os.path.join(self.test_data_root, self.videos_path,
                                            folder))
            for item in files:
                if item.endswith(".mp4") or item.endswith(".3gpp"):
                    os.remove(os.path.join(self.test_data_root, self.videos_path,
                                           folder, item))

    def test_grab_from_video(self):
        logtime_data = {}
        self.data_set.grap_from_video(path=os.path.join(self.test_data_root,
                                                        self.videos_path),
                                      log_time=logtime_data)

        print("Time used to grap samples from video is ",
              logtime_data.get('GRAP_FROM_VIDEO'), "ms")
        self.assertGreater(logtime_data.get('GRAP_FROM_VIDEO'), 0), 'No videos re ' \
                                                                    'grabbed '

    def test_grap_from_disk(self):
        logtime_data = {}
        self.data_set.grap_from_disk(sample_folder=os.path.join(self.test_data_root,
                                                                "sample_pickles"
                                                                "/positiveSamples"),
                                     log_time=logtime_data)
        print("Time used to grap samples from disk is ",
              logtime_data.get('GRAP_FROM_DISK'), "ms")
        self.assertGreater(logtime_data.get('GRAP_FROM_DISK'), 0), 'No videos were ' \
                                                                   'grabbed '


class TestSaveBalancedDataset(unittest.TestCase):
    def test_save_balanced_dataset(self):
        pass


class TestTransformations(unittest.TestCase):
    def test_transform_to_hdf5(self):
        rootDir = "test/unit-tests/data-and-models/testData"
        dSet.transform_to_hdf5(path=os.path.join(rootDir, "sample_pickles"),
                               hdf5_path=os.path.join(rootDir,
                                                      "sample_pickles"), testing=True)
        self.assertTrue(os.path.exists(
            os.path.join(rootDir, "sample_pickles/vvad_train.hdf5")) and
                        os.path.exists(os.path.join(rootDir,
                                                    "sample_pickles/vvad_validation"
                                                    ".hdf5")))
        os.remove(os.path.join(rootDir, "sample_pickles/vvad_train.hdf5"))
        os.remove(os.path.join(rootDir, "sample_pickles/vvad_validation.hdf5"))

    def test_transform_points_to_numpy(self):
        from types import SimpleNamespace
        points = [
            {'x': 0, 'y': 0.25},
            {'x': 1, 'y': 0.5},
            {'x': 2, 'y': 0.75},
            {'x': 3, 'y': 1.0},
            {'x': 4, 'y': 1.25}
        ]
        pointsNamespace = []
        for point in points:
            pointsNamespace.append(SimpleNamespace(**point))

        self.assertEqual(type(dSet.transform_points_to_numpy(pointsNamespace)),
                         numpy.ndarray), 'Argument of wrong type!'
        self.assertEqual(dSet.transform_points_to_numpy(pointsNamespace)[2][1],
                         pointsNamespace[2].y), "Array's content is not correct!"


class TestMakeTestSet(unittest.TestCase):
    def test_make_test_set(self):
        pass


if __name__ == "__main__":
    print("name", __name__)
    unittest.main(warnings='ignore')
