import unittest

import cv2
import imutils

import vvadlrs3.utils.imageUtils as imgUtils


class TestImageUtils(unittest.TestCase):
    """
        Test the detection of faces in an image
    """
    # Returns the bounding box for the first found face in an image, positive test case
    def test_detect_face(self):
        test_img_path = "./testData/imgUtils_image_example.jpg"

        self.assertNotEqual(False, imgUtils.get_random_face_from_img(
            load_local_img(test_img_path)))

    # Returns False as no face is existent in the example image
    def test_detect_no_face(self):
        test_img_path = "./testData/imgUtils_image_neg_example.jpg"

        self.assertEqual(False, imgUtils.get_random_face_from_img(
            load_local_img(test_img_path)))

    """
        Test the cropping of images
    """

    def test_crop_image(self):
        pass

    """
    """

    def test_to_image_space(self):
        pass

    """
    """

    def test_resize_and_zero_padding(self):
        test_img_path = "testData/imgUtils_image_example.jpg"
        shape = (120, 150)

        new_img = imgUtils.resize_and_zero_padding(load_local_img(test_img_path), shape)
        self.assertEqual(new_img.shape[0], shape[1])
        self.assertEqual(new_img.shape[1], shape[0])

    """
        Check if sample example is converted to video and exists in temp directory
    """

    def test_convert_sample_to_video(self):
        pass

    def test_convert_sample_to_video_fail(self):
        pass


def load_local_img(path_to_local_img):
    # load the input image from disk, resize it, and convert it from
    # BGR to RGB channel ordering (which is what dlib expects)
    image = cv2.imread(path_to_local_img)

    image = imutils.resize(image, width=200)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


if __name__ == '__main__':
    unittest.main()
