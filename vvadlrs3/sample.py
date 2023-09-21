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
import dlib
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
            init_pos (list of floats): A bounding box for the initial face. Relative or absolute pixel values in format (x, y, w, h)
            internal_rect_oversize (float): the percentage of which the initial
            relative (boolean): relative or absolute pixel values

        Returns:
            Nothing
        """
        if type(init_pos) == dlib.rectangle or type(init_pos) == dlib.drectangle:
            self.init_pos = (init_pos.tl_corner().x, init_pos.tl_corner(
            ).y, init_pos.width(), init_pos.height())
        else:
            self.init_pos = init_pos
        self.internal_rect_oversize = internal_rect_oversize
        self.tracker = None
        self.relative = relative

    def getNextFace(self, image):
        """Returns the next FaceImage and the pos of the face in the original image Space

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
                imageWidth = image.shape[1]
                imageHeight = image.shape[0]
                x = self.init_pos[0] * imageWidth
                y = self.init_pos[1] * imageHeight
                w = self.init_pos[2] * imageWidth
                h = self.init_pos[3] * imageHeight
            else:
                x = self.init_pos[0]
                y = self.init_pos[1]
                w = self.init_pos[2]
                h = self.init_pos[3]
        xStart = x * (1 - self.internal_rect_oversize)
        yStart = y * (1 - self.internal_rect_oversize)
        xEnd = (x + w) * 1.2
        yEnd = (y + h) * 1.2
        ROIrect = dlib.drectangle(xStart, yStart, xEnd, yEnd)
        ROI = cropImage(image, ROIrect)
        detector = dlib.get_frontal_face_detector()
        dets = detector(ROI, 1)
        # TODO: error if more than one face! - invalid
        if len(dets) != 1:
            print("Invalid Sample because there are {} faces".format(len(dets)))
            return False, False  # Means Error
        dInImage = toImageSpace(ROIrect, dets[0])
        face = cropImage(image, dInImage)

        if not self.tracker:
            self.tracker = dlib.correlation_tracker()
            self.tracker.start_track(image, dInImage)
        return face, dInImage


class FaceFeatureGenerator:
    """
    This class can generate the features for the different approaches.
    """

    def __init__(self, featureType, shapeModelPath=None, shape=None):
        """
        init for the specific featureType

        Args:
            featureType (String ["faceImage", "lipImage", "faceFeatures", "lipFeatures"]): type of the feature map that should be returned by getFeatures()
            shapeModelPath (String): path to the model for the shape_predictor

        Returns:
            Nothing
        """
        self.supportedFeatureTypes = [
            "faceImage", "lipImage", "faceFeatures", "lipFeatures", 'all', "allowfaceImage"]
        assert featureType in self.supportedFeatureTypes, "unsupported featureType {}. Supported featureTypes are {}".format(
            featureType, self.supportedFeatureTypes)
        if featureType == "faceImage":
            assert shape, "For featureType {} a shape must be set".format(
                featureType)
        else:
            assert shapeModelPath, "For featureType {} a shapeModelPath must be set".format(
                featureType)
            if featureType == "lipImage":
                assert shape, "For featureType {} a shape must be set".format(
                    featureType)
        self.shape = shape
        self.featureType = featureType
        self.predictor = dlib.shape_predictor(shapeModelPath)

    def getFeatures(self, image):
        """
        generates a feature map of the type given in the constructor

        Args:
            image (image): openCV image in RGB format

        Returns:
            feature (depends): Returns feature map depending on given feature Type (faceImage, lipImage, faceFeature, lipFeatures, all)
        """
        if self.featureType == "faceImage":
            return resizeAndZeroPadding(image, self.shape)
        elif self.featureType == "faceFeatures":
            shape = self.predictor(image, dlib.rectangle(
                0, 0, image.shape[1], image.shape[0]))
            return shape.parts()
        elif self.featureType == "lipFeatures":
            shape = self.predictor(image, dlib.rectangle(
                0, 0, image.shape[1], image.shape[0]))
            lipShape = dlib.points()
            for i, point in enumerate(shape.parts()):
                if i > 47 and i < 68:
                    lipShape.append(point)
            return lipShape
        elif self.featureType == "lipImage":
            shape = self.predictor(image, dlib.rectangle(
                0, 0, image.shape[1], image.shape[0]))
            lipShape = dlib.points()
            for i, point in enumerate(shape.parts()):
                if i > 47 and i < 68:
                    lipShape.append(point)
            lipXStart = min(lipShape, key=lambda p: p.x).x
            lipXEnd = max(lipShape, key=lambda p: p.x).x
            lipYStart = min(lipShape, key=lambda p: p.y).y
            lipYEnd = max(lipShape, key=lambda p: p.y).y
            lipRect = dlib.drectangle(lipXStart, lipYStart, lipXEnd, lipYEnd)
            return resizeAndZeroPadding(cropImage(image, lipRect), self.shape)
        elif self.featureType == "all":  # returns in decending order: faceImage, lipImage, faceFeature, lipFeatures
            shape = self.predictor(image, dlib.rectangle(
                0, 0, image.shape[1], image.shape[0]))
            faceFeatures = shape.parts()
            lipShape = dlib.points()
            for i, point in enumerate(shape.parts()):
                if i > 47 and i < 68:
                    lipShape.append(point)
            lipFeatures = lipShape
            lipXStart = min(lipShape, key=lambda p: p.x).x
            lipXEnd = max(lipShape, key=lambda p: p.x).x
            lipYStart = min(lipShape, key=lambda p: p.y).y
            lipYEnd = max(lipShape, key=lambda p: p.y).y
            lipRect = dlib.drectangle(lipXStart, lipYStart, lipXEnd, lipYEnd)
            lipImage = resizeAndZeroPadding(
                cropImage(image, lipRect), self.shape)
            faceImage = resizeAndZeroPadding(image, self.shape)
            return faceImage, lipImage, faceFeatures, lipFeatures
        # returns in descending order without faceImage: lipImage, faceFeature, lipFeatures
        elif self.featureType == "allowfaceImage":
            shape = self.predictor(image, dlib.rectangle(
                0, 0, image.shape[1], image.shape[0]))
            faceFeatures = shape.parts()
            lipShape = dlib.points()
            for i, point in enumerate(shape.parts()):
                if 47 < i < 68:
                    lipShape.append(point)
            lipFeatures = lipShape
            lipXStart = min(lipShape, key=lambda p: p.x).x
            lipXEnd = max(lipShape, key=lambda p: p.x).x
            lipYStart = min(lipShape, key=lambda p: p.y).y
            lipYEnd = max(lipShape, key=lambda p: p.y).y
            lipRect = dlib.drectangle(lipXStart, lipYStart, lipXEnd, lipYEnd)
            lipImage = resizeAndZeroPadding(
                cropImage(image, lipRect), self.shape)
            return lipImage, faceFeatures, lipFeatures
        else:
            # should never happen
            raise AttributeError("unsupported featureType {}. Supported featureTypes are {}".format(
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
            label (boolean): positive or negative Label
            type (String out of ["faceImages", "mouthImages", "faceFeatures", "mouthFeatures"]): the type of this sample
            for the specific approach
            shape (Tuple of ints): the shape to which an Image should be scaled and zeroPadded
        """
        self.data = []
        self.label = None
        self.featureType = None
        self.shape = None
        self.k = None

    def isValid(self):
        return len(self.data) == self.k

    # @timeit
    def getData(self, imageSize=None, num_steps=None, grayscale=False, normalize=False):
        # TODO: Add description
        """ DESCRIPTION MISSING

        Args:
            imageSize (Tuple of ints): size of the sample's images
            num_steps (int): number of steps for the sample
            grayscale (boolean): decides wheater to use grayscale images or not

        Returns:
            feature_map (numpy array): Return the feature map as a numpy array
        """
        # TODO assert imageSize is quadratic - Nope! not for lipImages - cv2.resize wont work - mayberesizeandpadding??
        if num_steps and num_steps < self.k:
            stepData = self.data[:num_steps]
        else:
            stepData = self.data

        if imageSize:
            shape = list(stepData.shape)
            shape[1] = imageSize[0]
            shape[2] = imageSize[1]
            if grayscale:
                shape = shape[:-1]
            imageData = np.empty(shape, dtype=stepData.dtype)
            for i, image in enumerate(stepData):
                image = cv2.resize(image, imageSize)
                if grayscale:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                imageData[i] = image

        else:
            imageData = stepData
        if normalize:
            assert "Features" in self.featureType, "Normalize is only possible for face and lipFeatures"
            # calc euclidean dist vector
            # this sets the dtype to np.float64
            outputArray = self._getDist(imageData)
            outputArray = self._normalize(outputArray)
            # TODO: normalize using np.norm? - how is it working? - all frames should be normalized dependently while all samples should be normalized independently

        else:
            outputArray = imageData
        if num_steps == 1:
            outputArray = outputArray[0]

        return np.array(outputArray)

    def _getDist(self, sample):
        # ToDo check functionality and add RETURN value
        """ calculating the distance vectors for a sample

        Args:
            sample (numpy array): the sample we want the distances to be calculated

        Returns:
            outSample (unclear): Returns unclear data
        """
        outSample = np.empty(sample.shape)  # this sets the dtype to np.float64
        base = sample[0][0]

        for frame_num, frame in enumerate(sample):
            newFrame = np.empty(frame.shape)
            for pos_num, pos in enumerate(frame):
                # TODO: calculate distance to base
                xdist = pos[0] - base[0]
                ydist = pos[1] - base[1]
                newFrame[pos_num] = [xdist, ydist]
            outSample[frame_num] = newFrame
        return outSample

    def _normalize(self, arr):
        """ Normalizes the features of the array to [-1, 1].

            Args:
                arr (numpy array): array with features to normalize

            Returns:
                arr_norm (numpy array): numpy array with features normalized to [-1, 1]
        """
        arrMax = np.max(arr)
        arrMin = np.min(arr)
        absMax = np.max([np.abs(arrMax), np.abs(arrMin)])
        return arr / absMax

    def getLabel(self):
        """ returns the label as int

        Returns:
            label (int): Returns label as integer value
        """
        return int(self.label)

    def generateSampleFromFixedFrames(self, k, frames, init_pos, label, featureType, shape, shapeModelPath=None,
                                      relative=True):
        #ToDo add description
        """ Generates Sample from fixed frames without return value

        Args:
            k (int): defines the temporal sliding window in frames
            frames ():
            init_pos ():
            label (boolean): positive or negative Label
            featureType (String): type of the feature map that should be returned by getFeatures()
            shape (Tuple of ints): the shape to which an Image should be scaled and zeroPadded
            shapeModelPath (String): path to the model for the shape_predictor
            relative (boolean): ??
        """
        # assert len frames to k
        # track face from init_pos
        self.label = label
        self.k = k
        self.featureType = featureType
        ffg = FaceFeatureGenerator(
            featureType, shapeModelPath=shapeModelPath, shape=shape)
        tracker = FaceTracker(init_pos, relative=relative)
        for x, image in enumerate(frames):
            face, boundingBox = tracker.getNextFace(image)
            if boundingBox:  # check if tracking was successful
                self.data.append(ffg.getFeatures(face))
            else:
                print("did not get a face for frame number {}".format(x))
                break

    #KZ Find unused functions and delete them
    def generate_sample_from_buffer(self, sourcebuffer, k):
        """ get another frame from sourcebuffer

        Args:
            sourcebuffer ():
            k (int): defines the temporal sliding window in frames

        Returns:
            sample (any): returns False if sampleLength is smaller k otherwise returns a sample of length k

        """
        pass
        # is needed for live data and must be implemented
        # use a ringbuffer here
        # empty buffer if one frame is invalid(no face)

    def visualize(self, fps=25, saveTo=None, supplier="pyplot"):
        """ visualize the sample depending on the featureType

        Args:
            fps (int): Frames per second
            saveTo (String): Path to save the visualization to (Default: None)
            supplier (String): Selection between "pyplot" and "opencv"

        """
        if "Image" in self.featureType:
            rc('animation', html='html5')
            fig = plt.figure()
            borderSize = int(self.data.shape[1] / 8)
            value = [0, 255, 0] if self.label else [255, 0, 0]
            images = [[plt.imshow(
                cv2.copyMakeBorder(cv2.cvtColor(features, cv2.COLOR_BGR2RGB), top=borderSize, bottom=borderSize,
                                   left=borderSize, right=borderSize, borderType=cv2.BORDER_CONSTANT, value=value),
                animated=True)] for features in self.data]

            print("shape: {}".format(self.data.shape))

            if supplier == "pyplot":
                # images = [[plt.imshow(cv2.copyMakeBorder(cv2.cvtColor(features, cv2.COLOR_BGR2RGB), top=borderSize, bottom=borderSize, left=borderSize, right=borderSize, borderType=cv2.BORDER_CONSTANT, value=value), animated=True)] for features in self.data]
                ani = animation.ArtistAnimation(fig, images, interval=(1 / fps) * 1000, blit=True,
                                                repeat_delay=1000)
                if saveTo:
                    ani.save(saveTo, writer='imagemagick')
                plt.show()
            elif supplier == "opencv":
                for features in self.data:
                    time.sleep(1 / fps)
                    borderSize = 25
                    value = [0, 255, 0] if self.label else [0, 0, 255]
                    featuresWithBorder = cv2.copyMakeBorder(
                        features, top=borderSize, bottom=borderSize, left=borderSize, right=borderSize,
                        borderType=cv2.BORDER_CONSTANT, value=value)
                    cv2.imshow(self.featureType, featuresWithBorder)
                    key = cv2.waitKey(1) & 0xFF
                    # if the `q` key was pressed, break from the loop
                    if key == ord("q"):
                        break
                cv2.destroyAllWindows()
        else:
            imgRatio = 200
            data = self.getData(normalize=True)  # normalize=True
            # calc maximal imageSize for the values from the shape
            print("shape: {}".format(data.shape))
            max_x, max_y = np.max(np.amax(data, axis=1), axis=0)
            min_x, min_y = np.min(np.amin(data, axis=1), axis=0)
            print("Max_x: {}\nMax_y: {}".format(max_x, max_y))
            print("Min_x: {}\nMin_y: {}".format(min_x, min_y))
            # This does not work for lips because they are always in the lower section of the image which is higher values

            rc('animation', html='html5')
            fig = plt.figure()
            # This does not make to much sense here...
            borderSize = int(data.shape[1] / 8)
            value = [0, 255, 0] if self.label else [255, 0, 0]
            images = []
            # features = np.zeros((max_x + 2*borderSize, max_y+ 2*borderSize, 3), dtype=np.uint8)
            features = np.zeros(
                (imgRatio + 2 * borderSize, imgRatio + 2 * borderSize, 3), dtype=np.uint8)
            for frame in data:
                im = cv2.copyMakeBorder(cv2.cvtColor(features, cv2.COLOR_BGR2RGB), top=borderSize, bottom=borderSize,
                                        left=borderSize, right=borderSize, borderType=cv2.BORDER_CONSTANT, value=value)
                for x, y in frame:
                    cv2.circle(im, (int((x - min_x) * imgRatio),
                                    int((y - min_y) * imgRatio)), 1, (255, 255, 255), -1)
                images.append([plt.imshow(im, animated=True)])
            ani = animation.ArtistAnimation(fig, images, interval=(1 / fps) * 1000, blit=True,
                                            repeat_delay=1000)
            if saveTo:
                ani.save(saveTo, writer='imagemagick')
            plt.show()

    def save(self, path):
        """ saves the sample to a pickle file - data is converted to a numpyArray first

        Args:
            path (String): Path to save the file to
        """
        self.data = np.array(self.data)
        with open(path, 'wb') as file:
            pickle.dump(self.__dict__, file)

    def load(self, path):
        """ loads from a pickle file

        Args:
            path (String): Path to load the pickle file from
        """
        with open(path, 'rb') as file:
            self.__dict__.clear()
            self.__dict__.update(pickle.load(file))


# Currently not used in the pipeline, is a "nice to have" feature
def visualizeSamples(folder):
    """ visualize positive and negative samples from a folder.

    Args:
        folder (String): Path to folder where negative and positive samples are saved (positiveSamples/negativeSamples)
    """
    positiveFolder = os.path.join(folder, "positiveSamples")
    negativeFolder = os.path.join(folder, "negativeSamples")
    sampleFiles = []

    for file in glob.glob(os.path.join(positiveFolder, "*.pickle")):
        sampleFiles.append(file)
    for file in glob.glob(os.path.join(negativeFolder, "*.pickle")):
        sampleFiles.append(file)
    print(sampleFiles)
    # put whole path
    random.shuffle(sampleFiles)
    for sampleFile in sampleFiles:
        sample = FeaturedSample()
        sample.load(sampleFile)
        sample.visualize()
