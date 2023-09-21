import glob
import os.path
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from vvadlrs3 import sample as sample, pretrained_models
from vvadlrs3 import dataSet


def get_rgb_test_image(image_file_name, folder_path):
    img = cv2.imread(os.path.join(folder_path, image_file_name))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class TestFaceTracker(unittest.TestCase):

    def setUp(self):
        self.test_data_root = "test/unit-tests/data-and-models/testData"
        self.images_path = "images"
        self.face_tracker = sample.FaceTracker(
            init_pos=(0, 0, 1, 1)
        )

    def test_get_next_face(self):
        image_file = "One_human_face.jpg"
        images_path = os.path.join(self.test_data_root, self.images_path)
        RGBimg = get_rgb_test_image(image_file_name=image_file, folder_path=images_path)

        _, box = self.face_tracker.get_next_face(RGBimg)

        self.assertTrue(box)

    def test_get_next_face_relative(self):
        self.face_tracker.relative = True
        image_file = "One_human_face.jpg"
        images_path = os.path.join(self.test_data_root, self.images_path)
        RGBimg = get_rgb_test_image(image_file_name=image_file, folder_path=images_path)

        _, box = self.face_tracker.get_next_face(RGBimg)

        self.assertTrue(box)

    def test_get_next_face_fail(self):
        image_file = "Two_human_faces.jpg"
        images_path = os.path.join(self.test_data_root, self.images_path)
        RGBimg = get_rgb_test_image(image_file_name=image_file, folder_path=images_path)

        _, box = self.face_tracker.get_next_face(RGBimg)

        self.assertFalse(box)


class TestFaceFeatureGenerator(unittest.TestCase):
    def setUp(self):
        self.test_data_root = "test/unit-tests/data-and-models/testData"
        self.images_path = "images"

    def test_get_features_face_image(self):
        model = pretrained_models.get_face_img_model()  # model for predictions
        input_shape = model.layers[0].input_shape[2:]
        generator = sample.FaceFeatureGenerator(
            feature_type="faceImage",
            shape_model_path="models/shape_predictor_68_face_landmarks.dat",
            shape=(input_shape[1],
                   input_shape[0])
        )

        image_file = "One_human_face.jpg"
        images_path = os.path.join(self.test_data_root, self.images_path)
        RBGimg = get_rgb_test_image(image_file_name=image_file, folder_path=images_path)

        features = generator.get_features(image=RBGimg)

        self.assertIsNotNone(features)

    def test_get_features_lip_image(self):
        model = pretrained_models.get_lip_img_model()  # model for predictions
        input_shape = model.layers[0].input_shape[2:]
        generator = sample.FaceFeatureGenerator(
            feature_type="lipImage",
            shape_model_path="models/shape_predictor_68_face_landmarks.dat",
            shape=(input_shape[1],
                   input_shape[0])
        )

        image_file = "One_human_face.jpg"
        images_path = os.path.join(self.test_data_root, self.images_path)
        RBGimg = get_rgb_test_image(image_file_name=image_file, folder_path=images_path)

        features = generator.get_features(image=RBGimg)
        self.assertIsNotNone(features)

    def test_get_features_face_features(self):
        model = pretrained_models.get_face_feature_model()  # model for predictions
        input_shape = model.layers[0].input_shape[2:]
        generator = sample.FaceFeatureGenerator(
            feature_type="faceFeatures",
            shape_model_path="models/shape_predictor_68_face_landmarks.dat",
            shape=(input_shape[1],
                   input_shape[0])
        )

        image_file = "One_human_face.jpg"
        images_path = os.path.join(self.test_data_root, self.images_path)
        RBGimg = get_rgb_test_image(image_file_name=image_file, folder_path=images_path)

        features = generator.get_features(image=RBGimg)

        self.assertIsNotNone(features)

    def test_get_features_lip_features(self):
        model = pretrained_models.get_lip_feature_model()  # model for predictions
        input_shape = model.layers[0].input_shape[2:]
        generator = sample.FaceFeatureGenerator(
            feature_type="lipFeatures",
            shape_model_path="models/shape_predictor_68_face_landmarks.dat",
            shape=(input_shape[1],
                   input_shape[0])
        )

        image_file = "One_human_face.jpg"
        images_path = os.path.join(self.test_data_root, self.images_path)
        RBGimg = get_rgb_test_image(image_file_name=image_file, folder_path=images_path)

        features = generator.get_features(image=RBGimg)

        self.assertIsNotNone(features)

    def test_get_features_all(self):
        model = pretrained_models.get_lip_feature_model()  # model for predictions
        input_shape = model.layers[0].input_shape[2:]
        generator = sample.FaceFeatureGenerator(
            feature_type="all",
            shape_model_path="models/shape_predictor_68_face_landmarks.dat",
            shape=(input_shape[1],
                   input_shape[0])
        )

        image_file = "One_human_face.jpg"
        images_path = os.path.join(self.test_data_root, self.images_path)
        RBGimg = get_rgb_test_image(image_file_name=image_file, folder_path=images_path)

        feature_face_img, feature_lip_img, feature_face_feat, feature_lip_feat = \
            generator.get_features(image=RBGimg)

        self.assertIsNotNone(feature_face_img)
        self.assertIsNotNone(feature_face_feat)
        self.assertIsNotNone(feature_lip_img)
        self.assertIsNotNone(feature_lip_feat)

    def test_get_features_all_without_face_feat(self):
        model = pretrained_models.get_lip_feature_model()  # model for predictions
        input_shape = model.layers[0].input_shape[2:]
        generator = sample.FaceFeatureGenerator(
            feature_type="allwfaceImage",
            shape_model_path="models/shape_predictor_68_face_landmarks.dat",
            shape=(input_shape[1],
                   input_shape[0])
        )

        image_file = "One_human_face.jpg"
        images_path = os.path.join(self.test_data_root, self.images_path)
        RBGimg = get_rgb_test_image(image_file_name=image_file, folder_path=images_path)

        lip_image, face_features, lip_features = generator.get_features(image=RBGimg)

        self.assertIsNotNone(lip_features)
        self.assertIsNotNone(face_features)
        self.assertIsNotNone(lip_image)

    # ToDo check why fails
    @unittest.expectedFailure
    def test_unsupported_feature(self):
        generator = sample.FaceFeatureGenerator(
            feature_type="lipFeature",
        )

        image_file = "One_human_face.jpg"
        images_path = os.path.join(self.test_data_root, self.images_path)
        RBGimg = get_rgb_test_image(image_file_name=image_file, folder_path=images_path)

        self.assertRaises(AssertionError,
                          lambda: generator.get_features(image=RBGimg))


class TestFeaturedSample(unittest.TestCase):
    def setUp(self):
        self.test_data_root = "test/unit-tests/data-and-models/testData"
        self.images_path = "images"
        self.sample_pickles = "sample_pickles"

    def test_is_valid(self):
        # Use arbitrary np array to do proof of concept
        test_sample = sample.FeatureizedSample()
        test_sample.data = [np.ones((2, 2))] * 5
        test_sample.k = 5
        self.assertTrue(test_sample.is_valid())
        test_sample.k = 2
        self.assertFalse(test_sample.is_valid())

    @unittest.expectedFailure
    def test_get_data(self):
        test_sample = sample.FeatureizedSample()
        all_pickles = glob.glob(self.test_data_root + "/sample_pickles/positiveSamples"
                                + '/**/*.pickle', recursive=True)
        print("Sample data: ", all_pickles[0])

        # transform feature
        dataSet.transform_to_features(
            path=all_pickles[0],
            shape_model_path="../../../models/shape_predictor_5_face_landmarks.dat ",
            shape=(200, 200))

        test_sample.load(all_pickles[0])
        print(test_sample.get_data(image_size=(200, 200), num_steps=2, grayscale=True,
                                   normalize=True))

    @unittest.expectedFailure
    def test_get_dist(self):
        test_sample = sample.FeatureizedSample()
        all_pickles = glob.glob(self.test_data_root + "/sample_pickles/positiveSamples"
                                + '/**/*.pickle', recursive=True)
        test_sample.load(all_pickles[0])
        print(test_sample.get_label())
        print(test_sample.k)
        print(test_sample.featureType)
        print(test_sample._get_dist(test_sample.data))


    def test_normalize(self):
        test_sample = sample.FeatureizedSample()

        array = np.arange(4).reshape((2, 2))
        normalized_array = test_sample._normalize(self, arr=array)
        self.assertEqual(np.max(normalized_array), 1)

        for row in normalized_array:
            for item in row:
                self.assertLessEqual(item, 1)

    def test_get_label(self):
        s = sample.FeatureizedSample()
        negative_sample_name = "testNegativeSample.pickle"
        sample_path = os.path.join(self.test_data_root, self.sample_pickles,
                                   negative_sample_name)
        s.load(sample_path)

        self.assertEqual(0, s.get_label())

        positive_sample_name = "testPositiveSample.pickle"
        sample_path = os.path.join(self.test_data_root, self.sample_pickles,
                                   positive_sample_name)
        s.load(sample_path)

        self.assertEqual(1, s.get_label())

    def test_generate_sample_from_fixed_frames(self):
        pass

    def test_generate_sample_from_buffer(self):
        pass

    def test_save_data_as_pickle(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            temp_dir = Path(tmpdirname)
            data_dir = temp_dir.joinpath("test.pickle")
            test_sample = sample.FeatureizedSample()
            test_sample.save(data_dir)

            self.assertEqual(data_dir.exists(), True)
            # ToDo check if file is empty

    def test_load_existing_pickle(self):
        # ToDo: Check exception type
        sample_path = os.path.join(self.test_data_root, self.sample_pickles,
                                   "testPositiveSample.pickle")
        try:
            test_sample = sample.FeatureizedSample()
            test_sample.load(sample_path)
        except ValueError:
            self.fail("ValueError raised unexpectedly!")


if __name__ == "__main__":
    print("name", __name__)
    unittest.main(warnings='ignore')
