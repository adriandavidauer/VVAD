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

    def test_analyze_video_check_fps(self):
        self.assertEqual(vidUtils.analyze_video(
            "testData/videoUtils_example.mp4").get("fps"), 25)
        # default: faceImage

    def test_analyze_video_face_features(self):
        self.assertEqual(
            vidUtils.analyze_video(
                "testData/videoUtils_example.mp4", "faceFeatures").get("feature_type"),
            "faceFeatures")

    def test_analyze_video_wrong_feature_type(self):
        vidUtils.analyze_video("testData/videoUtils_example.mp4", "wrongFeature")
        self.assertRaises(
            'feature_type must be one of ["faceImage", "lipImage", "faceFeatures", '
            '"lipFeatures"]')

    def test_analyze_video_check_path(self):
        self.assertEqual(
            vidUtils.analyze_video(
                "testData/videoUtils_example.mp4", "faceFeatures").get("video_path"),
            "./testData/videoUtils_example.mp4")
