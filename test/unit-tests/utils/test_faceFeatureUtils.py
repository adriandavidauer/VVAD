import unittest
import numpy as np

import wildvvad.utils.faceFeatureUtils as fu


class TestFeatureUtils(unittest.TestCase):
    """
    Test all Utility Features
    """

    def test_translate_point_cloud(self):
        """
            Unit test on the linear translation of a 3 dimensional point cloud.
            All entities of that point cloud should be tranlated by the translation
            vector.
        """

        point_cloud = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        translation_vector = np.array([1, 1, 1])
        expected_result = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        result = fu.translate_point_cloud(point_cloud, translation_vector)

        # Evaluate
        self.assertTrue(np.array_equal(result, expected_result))

    def test_rotate_point_cloud_matrix(self):
        """
            Unit test on the rotation of a given 3 dimensional point cloud matrix.
            The rotation is around a three-dimensional rotation matrix.
        """

        point_cloud = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        rotation_matrix = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        expected_result = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        result = fu.rotate_point_cloud_matrix(point_cloud, rotation_matrix)
        self.assertTrue(np.array_equal(result, expected_result))

    def test_compute_centroid(self):
        """
            Unit test of calculation of the centroid of a 3 dimensional point cloud
            matrix.
        """

        points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        expected_result = np.array([4, 5, 6])
        result = fu.compute_centroid(points)
        self.assertTrue(np.array_equal(result, expected_result))

    def test_compute_rotation_angle(self):
        """
            Unit test on the computation of the rotation angle of a line (formed by
            multiple points) on the x-y-plane
        """

        nose_bridge_landmarks = np.array([[0, 0], [1, 1]])
        expected_result = np.pi / 4  # 45 degrees in radians
        result = fu.compute_rotation_angle(nose_bridge_landmarks)
        self.assertAlmostEqual(result, expected_result)

    def test_rotate_point_cloud_angle(self):
        """
            Unit test on the rotation of a 3 dimensional point cloud around an angle
            in the x-y-plane.
        """

        points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        angle = np.pi / 2  # 90 degrees in radians
        expected_result = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        result = fu.rotate_point_cloud_angle(points, angle)
        self.assertTrue(np.array_equal(np.round(result), expected_result))

    def test_rotation_90_degrees_around_z_axis(self):
        """
        Test rotation of 90 degrees around the z-axis.
        The expected rotation matrix for this case is:
        [[ 0, -1,  0],
         [ 1,  0,  0],
         [ 0,  0,  1]]
        """
        axis = np.array([0, 0, 1])
        theta = - np.pi / 2  # -90 degrees in radians
        expected_rotation_matrix = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        result = fu.calculate_rotation_matrix(axis, theta)
        np.testing.assert_array_almost_equal(result, expected_rotation_matrix,
                                             decimal=6)

    def test_rotation_180_degrees_around_y_axis(self):
        """
        Test rotation of 180 degrees around the y-axis.
        The expected rotation matrix for this case is:
        [[-1,  0,  0],
         [ 0,  1,  0],
         [ 0,  0, -1]]
        """
        axis = np.array([0, 1, 0])
        theta = np.pi  # 180 degrees in radians
        expected_rotation_matrix = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ])
        result = fu.calculate_rotation_matrix(axis, theta)
        np.testing.assert_array_almost_equal(result, expected_rotation_matrix,
                                             decimal=6)

    def test_align_face_to_z_axis(self):
        """
        Test aligning a synthetic point cloud face to the z-axis.
        The point cloud should be aligned such that the eyes and chin are positioned
        correctly along the z-axis.
        """
        # Create a synthetic point cloud representing a face
        point_cloud = np.zeros((68, 3))

        # Set arbitrary points for right and left eye corners
        point_cloud[36] = [1, -1, 0]
        point_cloud[45] = [1, 1, 0]

        # Set arbitrary point for chin
        point_cloud[8] = [0, -3, 0]  # Adjusted to ensure the chin point is not co-linear with the eyes

        aligned_point_cloud = fu.align_face_to_z_axis(point_cloud)

        # Check if the middle point between eyes is at (0, 0, 0)
        middle_points_between_eyes = (aligned_point_cloud[36] +
                                      aligned_point_cloud[45]) / 2.0
        np.testing.assert_array_almost_equal(middle_points_between_eyes,
                                             [0, 0, 0], decimal=6)

        # Check if the eyes are aligned along the x-axis
        np.testing.assert_array_almost_equal(aligned_point_cloud[36][1:],
                                             [-1, 0], decimal=1)
        np.testing.assert_array_almost_equal(aligned_point_cloud[45][1:],
                                             [1, 0], decimal=1)

        # Check if the chin is aligned along the y-axis and z-axis
        self.assertAlmostEqual(aligned_point_cloud[8][0], 0, places=6)
        self.assertAlmostEqual(aligned_point_cloud[8][2], 0, places=6)


if __name__ == '__main__':
    unittest.main()
