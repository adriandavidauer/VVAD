import os
import unittest
import vvadlrs3.utils.videoUtils as vidUtils


class TestVideoUtils(unittest.TestCase):
    """
        Video file can be provided and the analyzer will return the following
            information:
            video_path: path to the video
            fps: the fps associated with the video
            feature_type: One out of ["faceImage", "lipImage", "faceFeatures",
            "lipFeatures"]
            frame_scores: dict of lists with the prediction of every frame.
                (A frame has k predictions if it is not in
                the beginning or end of the video because a sample has k frames and the
                samples overlap.)

    """

    def setUp(self):
        self.test_data_root = "test/unit-tests/utils/testData"

    def test_analyze_video_check_fps(self):
        self.assertEqual(vidUtils.analyze_video(
            self.test_data_root + "/videoUtils_example.mp4").get("fps"), 25)
        # default: faceImage

    def test_analyze_video_face_features(self):
        self.assertEqual(
            vidUtils.analyze_video(
                self.test_data_root + "/videoUtils_example.mp4",
                "faceFeatures").get("feature_type"),
            "faceFeatures")

    def test_analyze_video_lip_img(self):
        self.assertEqual(
            vidUtils.analyze_video(
                self.test_data_root + "/videoUtils_example.mp4",
                "lipImage").get("feature_type"),
            "lipImage")

    def test_analyze_video_lip_features(self):
        self.assertEqual(
            vidUtils.analyze_video(
                self.test_data_root + "/videoUtils_example.mp4",
                "lipFeatures").get("feature_type"),
            "lipFeatures")

    def test_analyze_video_wrong_feature_type(self):
        self.assertRaises(ValueError,
                          lambda: vidUtils.analyze_video(
                              self.test_data_root + "/videoUtils_example.mp4",
                              "wrongFeature"))

    def test_analyze_video_check_path(self):
        self.assertEqual(self.test_data_root + "/videoUtils_example.mp4",
                         vidUtils.analyze_video(
                             self.test_data_root + "/videoUtils_example.mp4",
                             "faceFeatures").get(
                             "video_path")
                         )

    def test_analyze_video_save_json(self):
        vidUtils.analyze_video(
            self.test_data_root + "/videoUtils_example.mp4",
            "faceFeatures",
            self.test_data_root + "/video_analysis_result.json")

        self.assertTrue(os.path.exists(
            self.test_data_root + "/video_analysis_result.json"))

        os.remove(self.test_data_root + "/video_analysis_result.json")


if __name__ == '__main__':
    unittest.main(warnings='ignore')
    