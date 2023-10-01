"""
This Module handles everything related to a sample.
"""
import glob
# System imports
import os
import pickle
# from collections import deque
import random

# 3rd party imports
import matplotlib.pyplot as plt
from matplotlib import animation, rc

# local imports
from vvadlrs3.utils.imageUtils import *
from vvadlrs3.utils.timeUtils import *


class FaceTracker:
    """
    tracks faces in Images
    """

    def __init__(self, init_pos, internal_rect_oversize=0.2, relative=True):
        """ initialize the tracker with an initial position of the face in the Image

        Args:
            init_pos (list of floats): A bounding box for the initial face. Relative or
                absolute pixel values in format (x, y, w, h)
            internal_rect_oversize (float): the percentage of which the initial
            relative (bool): relative or absolute pixel values
        """
        if type(init_pos) == dlib.rectangle or type(init_pos) == dlib.drectangle:
            self.init_pos = (init_pos.tl_corner().x, init_pos.tl_corner(
            ).y, init_pos.width(), init_pos.height())
        else:
            self.init_pos = init_pos
        self.internal_rect_oversize = internal_rect_oversize
        self.tracker = None
        self.relative = relative

    def get_next_face(self, image):
        """
        Returns the next FaceImage and the pos of the face in the original image Space

        Args:
            image (image): openCV image in RGB format

        Returns:
            face (image): next FaceImage
            dInImage (array of int): Position of the face in the original image space
        """
        if self.tracker:
            # get x,y, w,h from tracker
            self.tracker.update(image)
            pos = self.tracker.get_position()
            # unpack the position object
            # TODO: handle negative values
            x = int(pos.left())
            y = int(pos.top())
            w = int(pos.right()) - x
            h = int(pos.bottom()) - y

        else:
            if self.relative:
                # calculate absolute x,y,w,h from relative
                image_width = image.shape[1]
                image_height = image.shape[0]
                x = self.init_pos[0] * image_width
                y = self.init_pos[1] * image_height
                w = self.init_pos[2] * image_width
                h = self.init_pos[3] * image_height
            else:
                x = self.init_pos[0]
                y = self.init_pos[1]
                w = self.init_pos[2]
                h = self.init_pos[3]

        x_start = x * (1 - self.internal_rect_oversize)
        y_start = y * (1 - self.internal_rect_oversize)
        x_end = (x + w) * 1.2
        y_end = (y + h) * 1.2
        roi_rect = dlib.drectangle(x_start, y_start, x_end, y_end)
        roi = crop_img(image, roi_rect)

        detector = dlib.get_frontal_face_detector()
        dets = detector(roi, 1)
        print("Detected faces:", dets)
        # TODO: error if more than one face! - invalid
        if len(dets) != 1:
            print("Invalid Sample because there are {} faces".format(len(dets)))
            return False, False  # Means Error
        d_in_image = to_img_space(roi_rect, dets[0])
        face = crop_img(image, d_in_image)

        if not self.tracker:
            self.tracker = dlib.correlation_tracker()
            self.tracker.start_track(image, d_in_image)
        return face, d_in_image


class FaceFeatureGenerator:
    """
    This class can generate the features for the different approaches.
    """

    def __init__(self, feature_type, shape_model_path=None, shape=None):
        """
        init for the specific featureType

        Args:
            feature_type (String ["faceImage", "lipImage", "faceFeatures",
                "lipFeatures"]): type of the feature map that should be returned by
                getFeatures()
            shape_model_path (str): path to the model for the shape_predictor

        Returns:
            Nothing
        """
        self.supportedFeatureTypes = [
            "faceImage", "lipImage", "faceFeatures", "lipFeatures", 'all',
            "allwfaceImage"]
        assert feature_type in self.supportedFeatureTypes, \
            "unsupported featureType {}. Supported featureTypes are {}". \
            format(feature_type, self.supportedFeatureTypes)
        if feature_type == "faceImage":
            assert shape, "For featureType {} a shape must be set".format(
                feature_type)
        else:
            assert shape_model_path, "For featureType {} a shapeModelPath " \
                                     "must be set".format(feature_type)
            if feature_type == "lipImage":
                assert shape, "For featureType {} a shape must be set".format(
                    feature_type)
        self.shape = shape
        self.featureType = feature_type
        self.predictor = dlib.shape_predictor(shape_model_path)

    def get_features(self, image):
        """
        generates a feature map of the type given in the constructor

        Args:
            image (image): openCV image in RGB format

        Returns:
            feature (depends): Returns feature map depending on given feature Type
                (faceImage, lipImage, faceFeature, lipFeatures, all)
        """
        if self.featureType == "faceImage":
            return resize_and_zero_padding(image, self.shape)
        elif self.featureType == "faceFeatures":
            shape = self.predictor(image, dlib.rectangle(
                0, 0, image.shape[1], image.shape[0]))
            return shape.parts()
        elif self.featureType == "lipFeatures":
            shape = self.predictor(image, dlib.rectangle(
                0, 0, image.shape[1], image.shape[0]))
            lip_shape = dlib.points()
            for i, point in enumerate(shape.parts()):
                if 47 < i < 68:
                    lip_shape.append(point)
            return lip_shape
        elif self.featureType == "lipImage":
            shape = self.predictor(image, dlib.rectangle(
                0, 0, image.shape[1], image.shape[0]))
            lip_shape = dlib.points()
            for i, point in enumerate(shape.parts()):
                if 47 < i < 68:
                    lip_shape.append(point)
            lip_x_start = min(lip_shape, key=lambda p: p.x).x
            lip_x_end = max(lip_shape, key=lambda p: p.x).x
            lip_y_start = min(lip_shape, key=lambda p: p.y).y
            lip_y_end = max(lip_shape, key=lambda p: p.y).y
            lip_rect = dlib.drectangle(lip_x_start, lip_y_start, lip_x_end, lip_y_end)
            return resize_and_zero_padding(crop_img(image, lip_rect), self.shape)
        elif self.featureType == "all":  # returns in descending order: faceImage,
            # lipImage, faceFeature, lipFeatures
            shape = self.predictor(image, dlib.rectangle(
                0, 0, image.shape[1], image.shape[0]))
            face_features = shape.parts()
            lip_shape = dlib.points()
            for i, point in enumerate(shape.parts()):
                if 47 < i < 68:
                    lip_shape.append(point)
            lip_features = lip_shape
            lip_x_start = min(lip_shape, key=lambda p: p.x).x
            lip_x_end = max(lip_shape, key=lambda p: p.x).x
            lip_y_start = min(lip_shape, key=lambda p: p.y).y
            lip_y_end = max(lip_shape, key=lambda p: p.y).y
            lip_rect = dlib.drectangle(lip_x_start, lip_y_start, lip_x_end, lip_y_end)
            lip_image = resize_and_zero_padding(
                crop_img(image, lip_rect), self.shape)
            face_image = resize_and_zero_padding(image, self.shape)
            return face_image, lip_image, face_features, lip_features
        # returns in decending order without faceImage: lipImage, faceFeature,
        # lipFeatures
        elif self.featureType == "allwfaceImage":
            shape = self.predictor(image, dlib.rectangle(
                0, 0, image.shape[1], image.shape[0]))
            face_features = shape.parts()
            lip_shape = dlib.points()
            for i, point in enumerate(shape.parts()):
                if 47 < i < 68:
                    lip_shape.append(point)
            lip_features = lip_shape
            lip_x_start = min(lip_shape, key=lambda p: p.x).x
            lip_x_end = max(lip_shape, key=lambda p: p.x).x
            lip_y_start = min(lip_shape, key=lambda p: p.y).y
            lip_y_end = max(lip_shape, key=lambda p: p.y).y
            lip_rect = dlib.drectangle(lip_x_start, lip_y_start, lip_x_end, lip_y_end)
            lip_image = resize_and_zero_padding(
                crop_img(image, lip_rect), self.shape)
            return lip_image, face_features, lip_features
        else:
            # should never happen
            raise AttributeError("unsupported featureType {}. Supported "
                                 "featureTypes are {}".format(
                self.featureType, self.supportedFeatureTypes))


class FeaturedSample:
    """
    This class represents a Sample(with the features for one specific approach)
    """

    def __init__(self):
        """
        init

            k (int): defines the temporal sliding window in frames
            data (List of numpy arrays): A list of featureVectors for this sample
            label (bool): positive or negative Label
            type (String out of ["faceImages", "mouthImages", "faceFeatures",
                "mouthFeatures"]): the type of this sample
            for the specific approach
            shape (Tuple of ints): the shape to which an Image should be scaled
                and zeroPadded
        """
        self.data = []
        self.label = None
        self.featureType = None
        self.shape = None
        self.k = None

    def is_valid(self):
        return len(self.data) == self.k

    # @timeit
    def get_data(self, image_size=None, num_steps=None, grayscale=False,
                 normalize=False):
        """
        Get image data as numpy arrays from loaded sample

        Args:
            imageSize (Tuple of ints): size of the sample's images
            num_steps (int): number of steps for the sample
            grayscale (bool): decides whether to use grayscale images or not

        Returns:
            feature_map (numpy array): Return the feature map as a numpy array
        """
        # TODO assert imageSize is quadratic - Nope! not for lipImages -
        if num_steps and num_steps < self.k:
            step_data = self.data[:num_steps]
        else:
            step_data = self.data

        if image_size:
            shape = list(step_data.shape)
            shape[1] = image_size[0]
            shape[2] = image_size[1]
            if grayscale:
                shape = shape[:-1]
            image_data = np.empty(shape, dtype=step_data.dtype)
            for i, image in enumerate(step_data):
                image = cv2.resize(image, image_size)
                if grayscale:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image_data[i] = image

        else:
            image_data = step_data
        if normalize:
            assert "Features" in self.featureType, "Normalize is only possible for " \
                                                   "face and lipFeatures"
            # calc euclidean dist vector
            # this sets the dtype to np.float64
            output_array = self._get_dist(image_data)
            output_array = self._normalize(output_array)
            # TODO: normalize using np.norm? - how is it working? - all frames should be
            #  normalized dependently while
            #  all samples should be normalized independently

        else:
            output_array = image_data
        if num_steps == 1:
            output_array = output_array[0]

        return np.array(output_array)

    @staticmethod
    def _get_dist(sample):
        """
        calculating the distance vectors for a sample

        Args:
            sample (numpy array): the sample we want the distances to be calculated

        Returns:
            outSample (unclear): Returns unclear data
        """
        out_sample = np.empty(sample.shape)  # this sets the dtype to np.float64
        base = sample[0][0]

        for frame_num, frame in enumerate(sample):
            new_frame = np.empty(frame.shape)
            for pos_num, pos in enumerate(frame):
                # TODO: calculate distance to base
                xdist = pos[0] - base[0]
                ydist = pos[1] - base[1]
                new_frame[pos_num] = [xdist, ydist]
            out_sample[frame_num] = new_frame
        return out_sample

    @staticmethod
    def _normalize(self, arr):
        """ Normalizes the features of the array to [-1, 1].

            Args:
                arr (numpy array): array with features to normalize

            Returns:
                arr_norm (numpy array): numpy array with features normalized to [-1, 1]
        """
        arr_max = np.max(arr)
        arr_min = np.min(arr)
        abs_max = np.max([np.abs(arr_max), np.abs(arr_min)])
        return arr / abs_max

    def get_label(self):
        """
        returns the label as int
        """
        return int(self.label)

    def generate_sample_from_fixed_frames(self, k, frames, init_pos, label,
                                          feature_type, shape, shape_model_path,
                                          relative=True):
        """
        Generates Sample from fixed frames without return value

        Args:
            k (int): defines the temporal sliding window in frames
            frames ():
            init_pos ():
            label (bool): positive or negative Label

            feature_type (str): type of the feature map that should be returned by
                getFeatures()
            shape (Tuple of ints): the shape to which an Image should be scaled and
                zeroPadded
            shape_model_path (str): path to the model for the shape_predictor
            relative (bool): ??
        """
        # assert len frames to k
        # track face from init_pos
        self.label = label
        self.k = k
        self.featureType = feature_type
        ffg = FaceFeatureGenerator(
            feature_type, shape_model_path=shape_model_path, shape=shape)
        tracker = FaceTracker(init_pos, relative=relative)
        for x, image in enumerate(frames):
            face, bounding_box = tracker.get_next_face(image)
            if bounding_box:  # check if tracker was successfull
                self.data.append(ffg.get_features(face))
            else:
                print("did not get a face for frame number {}".format(x))
                break

    def generate_sample_from_buffer(self, sourcebuffer, k):
        """ get another frame from sourcebuffer

        Args:
            sourcebuffer ():
            k (int): defines the temporal sliding window in frames

        Returns:
            sample (any): returns False if sampleLength is smaller k otherwise returns
                a sample of length k
        """
        pass  # is only needed for live data...see if I go there
        # use a ringbuffer here
        # empty buffer if one frame is invalid(no face)

    def visualize(self, fps=25, save_to=None, supplier="pyplot"):
        """ visualize the sample depending on the featureType

        Args:
            fps (int): Frames per second
            saveTo (str): Path to save the visualization to (Default: None)
            supplier (str): Selection between "pyplot" and "opencv"

        """
        if "Image" in self.featureType:
            rc('animation', html='html5')
            fig = plt.figure()
            border_size = int(self.data.shape[1] / 8)
            value = [0, 255, 0] if self.label else [255, 0, 0]
            images = [[plt.imshow(
                cv2.copyMakeBorder(cv2.cvtColor(features, cv2.COLOR_BGR2RGB),
                                   top=border_size, bottom=border_size,
                                   left=border_size, right=border_size,
                                   borderType=cv2.BORDER_CONSTANT, value=value),
                animated=True)] for features in self.data]

            print("shape: {}".format(self.data.shape))

            if supplier == "pyplot":
                # images = [[plt.imshow(cv2.copyMakeBorder(cv2.cvtColor(features,
                # cv2.COLOR_BGR2RGB), top=borderSize,
                # bottom=borderSize, left=borderSize, right=borderSize,
                # borderType=cv2.BORDER_CONSTANT, value=value),
                # animated=True)] for features in self.data]
                ani = animation.ArtistAnimation(fig, images, interval=(1 / fps) * 1000,
                                                blit=True,
                                                repeat_delay=1000)
                if save_to:
                    ani.save(save_to, writer='imagemagick')
                plt.show()
            elif supplier == "opencv":
                for features in self.data:
                    time.sleep(1 / fps)
                    border_size = 25
                    value = [0, 255, 0] if self.label else [0, 0, 255]
                    features_with_border = cv2.copyMakeBorder(
                        features, top=border_size, bottom=border_size, left=border_size,
                        right=border_size,
                        borderType=cv2.BORDER_CONSTANT, value=value)
                    cv2.imshow(self.featureType, features_with_border)
                    key = cv2.waitKey(1) & 0xFF
                    # if the `q` key was pressed, break from the loop
                    if key == ord("q"):
                        break
                cv2.destroyAllWindows()
        else:
            img_ratio = 200
            data = self.get_data(normalize=True)  # normalize=True
            # calc maximal imageSize for the values from the shape
            print("shape: {}".format(data.shape))
            max_x, max_y = np.max(np.amax(data, axis=1), axis=0)
            min_x, min_y = np.min(np.amin(data, axis=1), axis=0)
            print("Max_x: {}\nMax_y: {}".format(max_x, max_y))
            print("Min_x: {}\nMin_y: {}".format(min_x, min_y))
            # This does not work for lips because they're always in the lower section
            # of image which is higher values

            rc('animation', html='html5')
            fig = plt.figure()
            # This does not make to much sense here...
            border_size = int(data.shape[1] / 8)
            value = [0, 255, 0] if self.label else [255, 0, 0]
            images = []
            # features = np.zeros((max_x + 2*borderSize, max_y+ 2*borderSize, 3),
            # dtype=np.uint8)
            features = np.zeros(
                (img_ratio + 2 * border_size, img_ratio + 2 * border_size, 3),
                dtype=np.uint8)
            for frame in data:
                im = cv2.copyMakeBorder(cv2.cvtColor(features, cv2.COLOR_BGR2RGB),
                                        top=border_size, bottom=border_size,
                                        left=border_size, right=border_size,
                                        borderType=cv2.BORDER_CONSTANT,
                                        value=value)
                for x, y in frame:
                    cv2.circle(im, (int((x - min_x) * img_ratio),
                                    int((y - min_y) * img_ratio)), 1, (255, 255, 255),
                               -1)
                images.append([plt.imshow(im, animated=True)])
            ani = animation.ArtistAnimation(fig, images, interval=(1 / fps) * 1000,
                                            blit=True,
                                            repeat_delay=1000)
            if save_to:
                ani.save(save_to, writer='imagemagick')
            plt.show()

    def save(self, path):
        """ saves the sample to a pickle file - data is converted to a numpyArray first

        Args:
            path (str): Path to save the file to
        """
        self.data = np.array(self.data)
        with open(path, 'wb') as file:
            pickle.dump(self.__dict__, file)

    def load(self, path):
        """ loads from a pickle file

        Args:
            path (str): Path to load the pickle file from
        """
        with open(path, 'rb') as file:
            self.__dict__.clear()
            self.__dict__.update(pickle.load(file))


def visualize_samples(folder):
    """ visualize positive and negative samples from a folder.

    Args:
        folder (str): Path to folder where negative and positive samples are saved
            (positiveSamples/negativeSamples)
    """
    positive_folder = os.path.join(folder, "positiveSamples")
    negative_folder = os.path.join(folder, "negativeSamples")
    sample_files = []

    for file in glob.glob(os.path.join(positive_folder, "*.pickle")):
        sample_files.append(file)
    for file in glob.glob(os.path.join(negative_folder, "*.pickle")):
        sample_files.append(file)
    print(sample_files)
    # put whole path
    random.shuffle(sample_files)
    for sampleFile in sample_files:
        sample = FeaturedSample()
        sample.load(sampleFile)
        sample.visualize()
