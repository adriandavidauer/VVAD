"""
This Module creates a dataset for the purpose of the visual speech detection system.
"""
# System imports
import os
import pathlib
import argparse
from multiprocessing import Process
import shutil
import time
import pickle
import glob
# from collections import deque
import multiprocessing
import random
from pathlib import Path, PurePath


# 3rd party imports
from pytube import YouTube
import cv2
from ffmpy import FFmpeg
import dlib
import numpy as np
from file_read_backwards import FileReadBackwards
import matplotlib.pyplot as plt
import yaml
import h5py


# local imports
from vvadlrs3.utils.imageUtils import *
from vvadlrs3.utils.multiprocessingUtils import *
from vvadlrs3.sample import *
from vvadlrs3.utils.timeUtils import *


# end file header
__author__ = "Adrian Lubitz"
__copyright__ = "Copyright (c)2017, Blackout Technologies"


class Sample():
    """
    DEPRECATED: use FeatureizedSample in the future
    This class represents a Sample(Kind of a raw sample which can be transformed into a more specific sample for different approaches)
    """

    def __init__(self, data, label, config, shapeModelPath):
        """
        FRAMERATE NEEDS TO BE ADJUSTED BEFORE!!!
        initilaize the sample with the data, label and the config(x, y, w, h of bounding box of the start frame)

        :param data: A list of frames for this sample
        :type data: List of Frames(numpyarrays)
        :param label: positive or negative Label
        :type label: bool
        :param config: a dict holding the values for the bounding box in the start frame(x, y, w, h) and the frameRate
        :type config: dict of ints
        """
        self.data = data
        self.label = label
        self.config = config
        self.shapeModelPath = shapeModelPath
        self.__faceImages = []
        self.__mouthImages = []
        self.__faceFeatures = []
        self.__mouthFeatures = []
        self.__faceTrajectory = []
        self.__classifier = None
        self.valid = False
        self.fps = self.config["fps"]
        if self.fps == 25:
            self.valid = True

    def isValid(self):
        """
        returns whether the Sample is valid or not
        """
        return self.valid

    def getData(self):
        """
        Return the data
        """
        return self.data

    def makeFeatures(self, classifier="HOG"):
        """
        DEPRECATED because time inefficent and not modular - can be used for analyzis, shouldnt be used later
        return the image of the face with a generator
        You can download a trained facial shape predictor from:
        (HOG)    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        (CNN)   http://dlib.net/files/mmod_human_face_detector.dat.bz2
        """
        # TODO: maybe make splitted version for different features for realtime capability
        self.__faceImages = []
        self.__mouthImages = []
        self.__faceFeatures = []
        self.__mouthFeatures = []
        self.__classifier = classifier
        if classifier == "HOG":
            predictor_path = pathlib.Path(os.path.join(
                self.shapeModelPath, "shape_predictor_68_face_landmarks.dat"))
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor(str(predictor_path))
            x = self.config["x"]
            y = self.config["y"]
            w = self.config["w"]
            h = self.config["h"]
            # print(self.config)
            # win = dlib.image_window()
            tracker = None
            for image in self.data:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if tracker:
                    # get x,y, w,h from tracker
                    tracker.update(image)
                    pos = tracker.get_position()
                    # unpack the position object
                    # TODO: handle negative values
                    x = int(pos.left())
                    y = int(pos.top())
                    w = int(pos.right()) - x
                    h = int(pos.bottom()) - y

                else:
                    # calculate absolute x,y,w,h from relative
                    imageWidth = image.shape[1]
                    imageHeight = image.shape[0]
                    x = x * imageWidth
                    y = y * imageHeight
                    w = w * imageWidth
                    h = h * imageHeight

                # enlarge the bounding box by 20percent to every side (if possible) because somtimes its too small
                xStart = x*0.8
                yStart = y*0.8
                xEnd = (x+w)*1.2
                yEnd = (y+h)*1.2
                ROIrect = dlib.drectangle(xStart, yStart, xEnd, yEnd)
                ROI = self.cropImage(image, ROIrect)

                # win.clear_overlay()
                # # win.set_image(ROI)
                # win.set_image(image)
                dets = detector(ROI, 1)
                # TODO: error if more than one face! - invalid
                if len(dets) != 1:
                    self.valid = False
                    print("Invalid Sample because there are {} faces".format(len(dets)))
                    return
                for k, d in enumerate(dets):
                    shape = predictor(ROI, d)
                    dInImage = self.toImageSpace(ROIrect, d)
                    # TODO: add cropped image (with d) and the features to the lists
                    self.__faceFeatures.append(shape.parts())
                    lipShape = dlib.points()
                    for i, point in enumerate(shape.parts()):
                        if i > 47 and i < 68:
                            lipShape.append(point)
                    self.__mouthFeatures.append(lipShape)
                    lipXStart = min(lipShape, key=lambda p: p.x).x
                    lipXEnd = max(lipShape, key=lambda p: p.x).x
                    lipYStart = min(lipShape, key=lambda p: p.y).y
                    lipYEnd = max(lipShape, key=lambda p: p.y).y
                    lipRect = dlib.drectangle(
                        lipXStart, lipYStart, lipXEnd, lipYEnd)
                    lipImage = self.cropImage(ROI, lipRect)
                    self.__mouthImages.append(
                        cv2.cvtColor(lipImage, cv2.COLOR_BGR2RGB))
                    faceImage = self.cropImage(image, dInImage)
                    self.__faceImages.append(
                        cv2.cvtColor(faceImage, cv2.COLOR_BGR2RGB))

                # win.add_overlay(self.toImageSpace(ROIrect, d))
                # win.add_overlay(ROIrect)
                # dlib.hit_enter_to_continue()

                tracker = dlib.correlation_tracker()
                tracker.start_track(image, dInImage)

    def cropImage(self, image, drect):
        """
        crop the image with a drect
        """
        # make rect from drect
        rect = dlib.rectangle(drect)
        #print("ROI shape: {}\nlipRect: {}".format(image.shape, rect))
        # crop with max(width, heigh) and min(0)
        maxX = image.shape[1] - 1
        maxY = image.shape[0] - 1
        return image[np.clip(rect.top(), 0, maxY):np.clip(rect.bottom(), 0, maxY), np.clip(rect.left(), 0, maxX):np.clip(rect.right(), 0, maxX)]

    def toImageSpace(self, ROIrect, rect):
        """
        returns a rectangle in the coordinate space of the whole image

        :param ROIrect: rectangle in Image Space, which contains the rect
        :type ROIrect: dlib.drectangle
        :param rect: rectangle in ROI Space
        :type ROI: dlib.drectangle
        """
        return dlib.drectangle(ROIrect.left() + rect.left(), ROIrect.top() + rect.top(), ROIrect.left() + rect.right(), ROIrect.top() + rect.bottom())

    def getMouthImages(self, classifier="HOG"):
        """
        return the image of the mouth
        """
        if self.__mouthImages and self.__classifier == classifier and self.isValid():
            return self.__mouthImages
        else:
            self.makeFeatures(classifier)
            if self.__mouthImages and self.__classifier == classifier and self.isValid():
                return self.__mouthImages
            else:
                raise Exception("Sample not valid")

    def getFaceFeatures(self, classifier="HOG"):
        """
        return the features of the face
        """
        if self.__faceFeatures and self.__classifier == classifier and self.isValid():
            return self.__faceFeatures
        else:
            self.makeFeatures(classifier)
            if self.__faceFeatures and self.__classifier == classifier and self.isValid():
                return self.__faceFeatures
            else:
                raise Exception("Sample not valid")

    def getMouthFeatures(self, classifier="HOG"):
        """
        return the features of the mouth
        """
        if self.__mouthFeatures and self.__classifier == classifier and self.isValid():
            return self.__mouthFeatures
        else:
            self.makeFeatures(classifier)
            if self.__mouthFeatures and self.__classifier == classifier and self.isValid():
                return self.__mouthFeatures
            else:
                raise Exception("Sample not valid")

    def getFaceImages(self, classifier="HOG"):
        """
        return the images of the face
        """
        if self.__faceImages and self.__classifier == classifier and self.isValid():
            return self.__faceImages
        else:
            self.makeFeatures(classifier)
            if self.__faceImages and self.__classifier == classifier and self.isValid():
                return self.__faceImages
            else:
                raise Exception("Sample not valid")

    def visualize(self, option="data"):
        """
        Visualize the sample of a video

        :param option: option can be of the following ["data", "faceImages", "mouthImages", "faceFeatures", "mouthFeatures"]
        :type boundingBox: String
        """
        if option not in ["data", "faceImages", "mouthImages", "faceFeatures", "mouthFeatures"]:
            raise Exception(
                'Option given to Sample.visualize needs to one of the following ["data", "faceImages", "mouthImages", "faceFeatures", "mouthFeatures"]')

        images = True
        try:
            if option == "data":
                data = self.getData()
            if option == "faceImages":
                data = self.getFaceImages()
            if option == "mouthImages":
                data = self.getMouthImages()
            if option == "faceFeatures":
                data = self.getFaceFeatures()
                images = False
            if option == "mouthFeatures":
                data = self.getMouthFeatures()
                images = False
        except:
            return

        if images:
            for img in data:
                #print (img)
                cv2.imshow('image', img)
                cv2.waitKey(int((1/self.fps)*1000))
        else:
            for featureVector in data:
                print(len(featureVector))
            # raise Exception("Not implemented yet!")

    def visualizeFaces(self):
        """
        Visualize the cropped face from the video
        """
        self.makeFeatures()
        if self.isValid():
            for face in self.__faceImages:
                cv2.imshow('Face', face)
                cv2.waitKey(int((1/self.fps)*1000))
        else:
            print("Sample is invalid")


class DataSet():
    """
    This class handles everything involved with the datasets.
    From creation and downloading over cleaning and balancing to converting and displaying.
    """

    # TODO: add path to Parameters
    def __init__(self, shapeModelPath, debug, sampleLength, maxPauseLength, shape, path, fps):
        """
        Just initializing an empty dataset
        """
        self.tempPath = None
        self.shapeModelPath = shapeModelPath
        self.debug = debug
        self.sampleLength = sampleLength
        self.maxPauseLength = maxPauseLength
        self.k = int(round(self.getFrameFromSecond(self.sampleLength)))
        self.path = path
        self.fps = fps
        self.shape = shape
        self.dropouts = 0

    def debugPrint(self, debugMsg):
        """
        printing debug message if debug is set.
        """
        if self.debug:
            print(debugMsg)

    def downloadLRS3SampleFromYoutube(self, path):
        """
        downloading corrosponding video data for the LRS3 dataset from youtube

        :param path: Path to a folder containing the txt files
        :type path: String
        """

        currentFolder = os.path.abspath(path)
        #print (currentFolder)
        # open folder and get a list of files
        try:
            files = list(os.walk(currentFolder, followlinks=True))[0][2]
        except:
            raise Exception(
                "Data folder is probably not mounted. Or you gave the wrong path.")
        files = [pathlib.Path(os.path.join(currentFolder, file))
                 for file in files]
        # get the RefField
        for file in files:
            if file.suffix == ".txt":
                textFile = open(file)
                # hat anscheinend noch ein return mit drinne
                ref = textFile.readlines()[2][7:].rstrip()
                # print(ref)
                break

        # Prep ref checking
        videoFileWithoutExtension = pathlib.Path(
            os.path.join(currentFolder, ref))
        # check if video is already there
        alreadyDownloaded = False
        for file in files:
            if file.suffix != ".txt":  # A video is there
                if ref in file.resolve().stem:   # Fully downloaded
                    print("Video already downloaded")
                    alreadyDownloaded = True
                else:
                    print("Resatarting download of unfinished video")
                    os.remove(file)
                break
        if not alreadyDownloaded:
            videoUrl = "https://www.youtube.com/watch?v={}".format(ref)
            print("starting to download video from {}".format(videoUrl))
            # download in a temp file (will be the title of the Video in youtube)
            self.tempPath = None

            def timeoutableDownload(videoUrl, currentFolder):
                self.tempPath = YouTube(
                    videoUrl).streams.first().download(currentFolder)
                self.tempPath = pathlib.Path(self.tempPath)
                # if ready rename the file to the real name(will be the ref)
                os.rename(self.tempPath, str(
                    videoFileWithoutExtension) + self.tempPath.resolve().suffix)
            p = Process(target=timeoutableDownload,
                        args=(videoUrl, currentFolder))
            p.start()
            p.join(600)
            if p.is_alive():
                print("Timeout for Download reached!")
                # Terminate
                p.terminate()
                p.join()

    # TODO add option if you want to use whats there or download if neccessary
    def getAllPSamples(self, path, **kwargs):
        """
        making all the samples from this folder.

        :param path: Path to the DataSet folder containing folders, which contain txt files. (For Example the pretrain folder)
        :type videoPath: String
        """
        folders = list(os.walk(path, followlinks=True))[0][1]
        folders.sort()
        for folder in folders:
            # Video file is the only not txt file
            currentFolder = os.path.abspath(os.path.join(path, folder))
            for sample in self.getPositiveSamples(currentFolder, **kwargs):
                yield sample
            self.debugPrint("[getAllPSamples] Folder {} done".format(folder))

    # TODO add option if you want to use whats there or download if neccessary
    def getAllSamples(self, featureType, path=None, relative=True, dryRun=False, showStatus=False, **kwargs):
        """
        making all the samples from this folder.

        :param path: Path to the DataSet folder containing folders, which contain txt files. (For Example the pretrain folder)
        :type videoPath: String
        """
        if showStatus:
            ts = time.perf_counter()
            self.debugPrint("[getAllPSamples] ###### Status:   0% done")
        if not path:
            path = self.path
        folders = list(os.walk(path, followlinks=True))[0][1]
        folders.sort()
        for i, folder in enumerate(folders):
            # Video file is the only not txt file
            currentFolder = os.path.abspath(os.path.join(path, folder))
            for sample in self.getSamples(currentFolder, featureType=featureType, relative=relative, dryRun=dryRun):
                yield sample
            self.debugPrint("[getAllPSamples] Folder {} done".format(folder))
            if showStatus:
                self.debugPrint("[getAllPSamples] ###### Status:   {}% done".format(
                    float(i)/len(folders) * 100))
                self.debugPrint("[getAllPSamples] ### Time elapsed: {} ms".format(
                    (time.perf_counter() - ts) * 1000))

    def convertAllFPS(self, path):
        """
        convertting all the fps from this folder.

        :param path: Path to the DataSet folder containing folders, which contain txt files. (For Example the pretrain folder)
        :type videoPath: String
        """
        folders = list(os.walk(path, followlinks=True))[0][1]
        folders.sort()
        for folder in folders:
            # Video file is the only not txt file
            currentFolder = os.path.abspath(os.path.join(path, folder))
            try:
                self.convertFPS(currentFolder)
            except FileNotFoundError as e:
                self.debugPrint(str(e) + "Skipping folder")

    def downloadLRS3(self, path):
        """
        downloading corrosponding video data for the LRS3 dataset from youtube and saving the faceFrames in the corrosponding folder

        :param path: Path to the DataSet folder containing folders, which contain txt files. (For Example the pretrain folder)
        :type path: String
        """

        # for folder in path call cutTedVideo - need to extract the Video File first
        folders = list(os.walk(path, followlinks=True))[0][1]
        folders.sort()
        for folder in folders:
            # Video file is the only not txt file
            currentFolder = os.path.abspath(os.path.join(path, folder))
            self.downloadLRS3SampleFromYoutube(currentFolder)

    def getTXTFiles(self, path):
        """
        Get all the txt files with a generator from the path
        """
        currentFolder = os.path.abspath(path)
        try:
            files = list(os.walk(currentFolder, followlinks=True))[0][2]
        except:
            raise Exception(
                "Data folder is probably not mounted. Or you gave the wrong path.")
        files = [pathlib.Path(os.path.join(currentFolder, file))
                 for file in files]
        for file in files:
            if file.suffix == ".txt":
                yield file

    def getPositiveSamples(self, path, dryRun=False):
        """
        Returning all positive samples from a Video with a generator

        :param path: Path to a folder containing the txt files
        :type path: String
        :param dryRun: With a dry run you will not really return samples, just a list of tuples with start and end time of the positive samples
        :type dryRun: boolean
        :returns: generator
        """
        try:
            videoPath = self.getVideoPathFromFolder(path)
        except FileNotFoundError as e:
            self.debugPrint(e)
            return []  # No Samples...sorry :/

        # self.debugPrint(videoPath)

        folder = os.path.dirname(videoPath)
        frameList = []  # list of configs    [startFrame, endFrame , x, y, w, h] x,y,w,h are relative pixels

        # for every txt file
        for textFile in self.getTXTFiles(folder):
            frameList.extend(self.getSampleConfigsForPositiveSamples(textFile))
            # firstFrameLine = ""
            # lastFrameLine = ""
            # textFile = open(textFile)
            # for line in textFile.readlines()[5:]:
            #     line = line.rstrip()
            #     if not firstFrameLine:# only the first line of the frames will be saved here
            #         firstFrameLine = line
            #     # if line is empty - last line of the frames
            #     if not line:
            #         break
            #     lastFrameLine = line
            # firstFrame = firstFrameLine.split()
            # lastFrame = lastFrameLine.split()
            #
            # configList = [int(firstFrame[0]), int(lastFrame[0]), float(firstFrame[1]), float(firstFrame[2]), float(firstFrame[3]), float(firstFrame[4])]
            # frameList.append(configList)

        frameList.sort(key=lambda x: x[0])

        # Open video
        if not dryRun:
            videoPath = self.convertFPS(videoPath.parents[0])
            vidObj = cv2.VideoCapture(str(videoPath))
            vidFps = vidObj.get(cv2.CAP_PROP_FPS)
        count = 0
        # sampleList = []
        for sampleConfig in frameList:
            if not self.checkSampleLength(self.getSecondFromFrame(sampleConfig[0]), self.getSecondFromFrame(sampleConfig[1])):
                continue
            if not dryRun:
                data = []
                label = True
                config = {"x": sampleConfig[2], "y": sampleConfig[3],
                          "w": sampleConfig[4], "h": sampleConfig[5], "fps": vidFps}
            # grap frames from start to endframe
                while(True):
                    success, image = vidObj.read()
                    if not success:
                        raise Exception(
                            "Couldnt grap frame of file {}".format(videoPath))
                    if(count >= sampleConfig[0] and count <= sampleConfig[1]):
                        data.append(image)
                    count += 1
                    if count > sampleConfig[1]:
                        break
                yield Sample(data, label, config, self.shapeModelPath)
            else:
                yield (self.getSecondFromFrame(sampleConfig[0]), self.getSecondFromFrame(sampleConfig[1]))

    def convertFPS(self, path, fps=25):
        """
        converting video in path to fps

        :param path: Path to a folder containing the txt files
        :type videoPath: String
        :param fps: frames per second
        :type fps: float
        """
        videoPath = self.getVideoPathFromFolder(path)
        folder = os.path.dirname(videoPath)

        vidObj = cv2.VideoCapture(str(videoPath))
        vidFps = vidObj.get(cv2.CAP_PROP_FPS)
        if vidFps != fps:
            # change the frameRate to 25, because the data set is expecting that!
            # ffmpeg -y -r 30 -i seeing_noaudio.mp4 -r 24 seeing.mp4
            oldVideoPath = videoPath
            videoPath = pathlib.Path(os.path.join(
                oldVideoPath.parents[0], oldVideoPath.stem + ".converted" + oldVideoPath.suffix))
            changeFps = FFmpeg(
                inputs={str(oldVideoPath): "-y"}, outputs={str(videoPath): '-r 25'})
            # print(changeFps.cmd)
            stdout, stderr = changeFps.run()
            # Remove the old!
            os.remove(oldVideoPath)
            self.debugPrint("Changed FPS of {} to {}".format(videoPath, fps))
        else:
            self.debugPrint("{} has already the correct fps".format(videoPath))
        return videoPath

    def getVideoPathFromFolder(self, path):
        """
        Get the path to the only video in folder raises Exception if there is none and removes all invalid files.
        """
        currentFolder = os.path.abspath(path)
        videoPath = None
        try:
            files = list(os.walk(currentFolder, followlinks=True))[0][2]
        except:
            raise Exception(
                "Data folder is probably not mounted. Or you gave the wrong path.")
        files = [pathlib.Path(os.path.join(currentFolder, file))
                 for file in files]
        videoFiles = []
        for file in files:
            if file.suffix not in [".txt"]:  # A video is there
                videoFiles.append(file)
        if not videoFiles:
            raise FileNotFoundError("No video in {}".format(currentFolder))
        if len(videoFiles) > 1:
            self.debugPrint(
                "TOO MANY VIDEOS OR OTHER FILES IN {}".format(currentFolder))
            for video in videoFiles:
                if ".converted" in video.stem:  # if there are two videos, dont remove the original!
                    self.debugPrint("Deleting {}".format(video))
                    os.remove(video)
                    return self.getVideoPathFromFolder(currentFolder)
        else:
            videoPath = videoFiles[0]

        return pathlib.Path(os.path.abspath(videoPath))

    def analyzeNegatives(self, path=None, saveTo=None):
        """
        Showing/Saving statistics over the data set.

        :param path: Path to the DataSet folder containing folders, which contain txt files. (For Example the pretrain folder)
        :type path: String
        """
        if not path:
            path = self.path
        folders = list(os.walk(path, followlinks=True))[0][1]
        folders.sort()
        numTotalSamples = 0
        pauses = []
        for folder in folders:
            currentFolder = os.path.abspath(os.path.join(path, folder))
            # for every txt file
            for textFile in self.getTXTFiles(currentFolder):
                numTotalSamples += 1
                videoPauses = self.getPauseLength(textFile)
                # if videoPauses:#
                #     print("VideoPauses: {}".format(videoPauses))
                pauses.extend(videoPauses)
            self.debugPrint("[analyzeNegatives] Folder {} done".format(folder))
        # TODO: norm to the number of analyzedSamples to see how many negative Samples can be constructed out of how many positive samples

        histData = [x[1] - x[0]
                    for x in pauses if self.checkSampleLength(x[0], x[1])]
        self.debugPrint(
            "Number of extracted positive samples:  {}".format(len(histData)))
        # bins=15)#np.arange(1.0, 19.0))
        plt.hist(histData, np.arange(self.sampleLength, max(histData)))
        plt.ylabel('Num Negatives', size=30)
        plt.xlabel('Sample Length', size=30)
        plt.xticks(size=30)
        plt.yticks(size=30)
        print("Total Amount of Samples is {}".format(numTotalSamples))
        if saveTo:
            plt.savefig(saveTo)
        else:
            plt.show()
        # return sorted(pauses, key = lambda x: x[0] - x[1])
        return pauses

    def analyzePositives(self, path, saveTo=None):
        """
        Showing/Saving statistics over the data set.

        :param path: Path to the DataSet folder containing folders, which contain txt files. (For Example the pretrain folder)
        :type path: String
        """
        pSamples = self.getAllPSamples(path, dryRun=True)
        # TODO: norm to the number of analyzedSamples to see how many negative Samples can be constructed out of how many positive samples

        histData = [x[1] - x[0]
                    for x in pSamples if self.checkSampleLength(x[0], x[1])]
        self.debugPrint(
            "Number of extracted positive samples:  {}".format(len(histData)))
        plt.hist(histData, np.arange(self.sampleLength, max(histData)))
        plt.ylabel('Num positive Samples')
        plt.xlabel('Sample Length')
        if saveTo:
            plt.savefig(saveTo)
        else:
            plt.show()
        # return sorted(pauses, key = lambda x: x[0] - x[1])
        return pSamples

    def getFrameFromSecond(self, second, fps=25):
        """
        calculates the frame in a video from a given second. (rounded off)

        :param second: second in the video
        :type second: float
        :param fps: framerate of video
        :type fps: float
        :returns: frame in video as float -> rounding needs to be made explicit
        """
        return float(second * fps)

    def getSecondFromFrame(self, frame, fps=25):
        """
        calculates the second in a video from a given frame.

        :param frame: frame in the video
        :type frame: float
        :param fps: framerate of video
        :type fps: float
        :returns: frame in video
        """
        return float(frame) / fps

    def getPauseLength(self, txtFile):
        """
        returns the length auf pauses and corrosponding start and end frame.

        :param txtFile: Path to the txt file
        :type txtFile: String
        :returns: List of pauses
        """
        lastStart = None
        pauses = []  # list of pauses defined by a tuple (startTime, endTime)
        with FileReadBackwards(txtFile) as txt:
            for l in txt:
                if "WORD START END ASDSCORE" in l:
                    break
                word, start, end, asdscore = l.split()
                # self.debugPrint("WORD: {} START: {} END: {} ASDSCORE: {}".format(word, start, end, asdscore))
                # end < lastStart:
                if lastStart and (float(lastStart) - float(end) > self.maxPauseLength):
                    # there is a pause
                    # if float(lastStart) - float(end) > 15.0:
                    #     print("Check out sample from {} where word is {}".format(txtFile, word))
                    pauses.append((float(end), float(lastStart)))
                    # self.debugPrint(pauses)
                lastStart = start
        # return sorted(pauses, key = lambda x: x[0] - x[1])
        return pauses

    def getSampleConfigsForPositiveSamples(self, txtFile):
        """
        returns a list of Frame configs for positive samples
        [startFrame, endFrame , x, y, w, h] x,y,w,h are relative pixels

        :param txtFile: Path to the txt file
        :type txtFile: String
        :returns: list of Frame configs
        """

        # check for Pauses
        pauses = self.getPauseLength(txtFile)
        # translate to Frames
        pauses = sorted([(int(np.ceil(self.getFrameFromSecond(x[0]))), int(
            self.getFrameFromSecond(x[1]))) for x in pauses], key=lambda x: x[0])

        # for all frames make a sample from start to pause0_start and from pause0_end to pause1_start ... pauseN_end to end
        firstFrameConfig = []
        lastFrameConfig = []
        pauseStart = []
        pauseEnd = []
        textFile = open(txtFile)
        configList = []
        for line in textFile.readlines()[5:]:
            currentConfig = line.rstrip().split()
            if not firstFrameConfig:  # only the first line of the frames will be saved here
                firstFrameConfig = currentConfig
                # add first frame num of the sample to all values of the pauses, because pauses are relative to start
                pauses = [(x[0] + int(firstFrameConfig[0]), x[1] +
                           int(firstFrameConfig[0])) for x in pauses]
            # check if the currentFrame is a pauseStart or pauseEnd frame
            if pauses:
                # self.debugPrint("currentConfig: {}\npauses: {}\nSample: {}".format(currentConfig, pauses, txtFile))
                if int(currentConfig[0]) in pauses[0]:
                    if not pauseStart and not pauseEnd:
                        # first sample
                        pauseStart = currentConfig
                        # from start to pauseStart
                        configList.append([int(firstFrameConfig[0]), int(pauseStart[0]), float(firstFrameConfig[1]), float(
                            firstFrameConfig[2]), float(firstFrameConfig[3]), float(firstFrameConfig[4])])
                    elif not pauseEnd and pauseStart:
                        # pauseStart is set so what I get is pauseEnd - just empty pauseStart and pop from pauses
                        pauseStart = []
                        pauses.pop(0)
                    elif not pauseStart and pauseEnd:
                        # pauseEnd is set so what I get is pauseStart - from pauseEnd to pauseStart and empty pauseEnd
                        configList.append([int(pauseEnd[0]), int(pauseStart[0]), float(
                            pauseEnd[1]), float(pauseEnd[2]), float(pauseEnd[3]), float(pauseEnd[4])])
                        pauseEnd = []
                    else:
                        # shouldnt happen!!
                        raise Exception("WTF")
            # if line is empty - last line of the frames
            if not line.rstrip():
                # put last sample
                # from pauseEnd to lastFrameConfig
                if pauses:
                    configList.append([int(pauseEnd[0]), int(lastFrameConfig[0]), float(
                        pauseEnd[1]), float(pauseEnd[2]), float(pauseEnd[3]), float(pauseEnd[4])])
                break
            lastFrameConfig = currentConfig

        assert len(
            pauses) == 0, "pauses is not empty...should be!\npauses: {}".format(pauses)
        if not configList:  # there where no pauses in this sample
            configList.append([int(firstFrameConfig[0]), int(lastFrameConfig[0]), float(firstFrameConfig[1]), float(
                firstFrameConfig[2]), float(firstFrameConfig[3]), float(firstFrameConfig[4])])
        return configList

    def checkSampleLength(self, start, end):
        """
        returns True if end - start is bigger than self.sampleLength

        :param start: start of the sample in seconds
        :type start: float
        :param end: end of the sample in seconds
        :type end: float
        :returns: boolean
        """
        return (end - start) > self.sampleLength

    def getSampleConfigs(self, txtFile):
        """
        returns a list of tuples holding the config of a sample consisting out of the following:
        [(label, [startFrame, endFrame , x, y, w, h]), ...] x,y,w,h are relative pixels of the bounding box in teh first frame

        :param txtFile: Path to the txt file
        :type txtFile: String
        :returns: list of tuples holding the frame config and corrosponding label
        """
        pauses = self.getPauseLength(txtFile)
        # translate to Frames
        pauses = sorted([(int(np.ceil(self.getFrameFromSecond(x[0]))), int(
            self.getFrameFromSecond(x[1]))) for x in pauses], key=lambda x: x[0])
        textFile = open(txtFile)
        configList = []
        negativeFrames = []
        counter = False
        for line in textFile.readlines()[5:]:
            if not line.rstrip():
                break
            currentConfig = line.rstrip().split()
            if not counter:  # only for the first line
                # initialize counter
                counter = int(currentConfig[0])
                # add first frame num of the sample to all values of the pauses, because pauses are relative to start
                pauses = [(x[0] + int(currentConfig[0]), x[1] +
                           int(currentConfig[0])) for x in pauses]
                # transform to list of all framenums associated with a pause/negative sample
                for pause in pauses:
                    negativeFrames.extend(list(range(pause[0], pause[1] + 1)))
                listLen = len(negativeFrames)
                # construct a set to check for the union
                negativeFrames = set(negativeFrames)
                assert listLen == len(
                    negativeFrames), "There are doublets in the frameList of pauses that shouldnt happen"
            assert counter >= int(currentConfig[0])
            # check if this frame needs to be taken in consideration
            if counter == int(currentConfig[0]):
                # check k frame numbers if they are in negativeFrames
                kFrameNumbers = set(list(range(counter, counter + self.k)))
                intersection = negativeFrames.intersection(kFrameNumbers)
                # check the three cases of the intersection
                if len(intersection) == 0:
                    # its a positive sample - save and skip k frames
                    sampleConfig = (True, [int(currentConfig[0]), int(currentConfig[0]) + (self.k-1), float(
                        currentConfig[1]), float(currentConfig[2]), float(currentConfig[3]), float(currentConfig[4])])
                    configList.append(sampleConfig)
                    counter += self.k
                elif len(intersection) == self.k:
                    # its a negative sample - save and skip k frames
                    sampleConfig = (False, [int(currentConfig[0]), int(currentConfig[0]) + (self.k-1), float(
                        currentConfig[1]), float(currentConfig[2]), float(currentConfig[3]), float(currentConfig[4])])
                    configList.append(sampleConfig)
                    counter += self.k
                else:
                    counter += 1  # its a mix of negative and positive frames -> just take the next k frames
        # check if last sample is valid (could reached out of video) counter is only allowed to be 1 frames bigger than currentConfig[0]
        if counter - int(currentConfig[0]) > 1:
            configList.pop()
        return configList

    def getSamples(self, path, featureType, shape, dryRun=False, relative=True):
        """
        Returning all samples from a Video with a generator

        :param path: Path to a folder containing the txt files
        :type path: String
        :param dryRun: With a dry run you will not really return samples, just a list of sampleConfigs
        :type dryRun: boolean
        :returns: generator
        """
        self.shape = shape
        try:
            videoPath = self.getVideoPathFromFolder(path)
        except FileNotFoundError as e:
            self.debugPrint(e)
            return []  # No Samples...sorry :/
        folder = os.path.dirname(videoPath)
        # list of configs    [startFrame, endFrame , x, y, w, h] x,y,w,h are relative pixels
        sampleConfigList = []
        # for every txt file
        for textFile in self.getTXTFiles(folder):
            sampleConfigList.extend(self.getSampleConfigs(textFile))
        sampleConfigList.sort(key=lambda x: x[1][0])
        if not dryRun:
            videoPath = self.convertFPS(videoPath.parents[0])
            vidObj = cv2.VideoCapture(str(videoPath))
            vidFps = vidObj.get(cv2.CAP_PROP_FPS)
            frames = []
            count = 0
            success = True
            while(success and sampleConfigList):
                # print("Sample from Frame {} to {}".format(sampleConfigList[0][1][0], sampleConfigList[0][1][1]))
                # print("Next Sample from Frame {} to {}".format(sampleConfigList[1][1][0], sampleConfigList[1][1][1]))
                # print("Counter: {}".format(count))
                if len(frames) == self.k:
                    sample = FeatureizedSample()
                    sample.generateSampleFromFixedFrames(
                        self.k, frames, sampleConfigList[0][1][2:], sampleConfigList[0][0], featureType=featureType, shape=self.shape, shapeModelPath=self.shapeModelPath)
                    if sample.isValid():
                        yield sample
                    else:
                        # TODO: keep track of the dropouts!
                        print("invalid sample")
                        self.dropouts += 1  # TODO: make threadsafe! https://docs.python.org/2/library/multiprocessing.html#sharing-state-between-processes
                    sampleConfigList.pop(0)
                    frames = []
                if sampleConfigList:
                    success, image = vidObj.read()
                    # TODO: Hier ist das Problem, dass die Liste leer gepopt wurde!!!!! BorderCase
                    if(count >= sampleConfigList[0][1][0] and count <= sampleConfigList[0][1][1]):
                        frames.append(image)
                        # print ("Added Frame. len(frames): {}".format(len(frames)))
                    count += 1
        else:
            for sample in sampleConfigList:
                yield sample

    def analyze(self, path=None, saveTo=None):
        """
        Shows statistics over the samples(values from the config, sum samples, negative samples, positive samples, ...)
        """
        numPositives = 0
        numNegatives = 0
        for sampleConfig in self.getAllSamples("faceImage", path, dryRun=True):
            if sampleConfig[0]:
                numPositives += 1
            else:
                numNegatives += 1

        labels = 'positive samples', 'negative samples'
        sizes = [numPositives, numNegatives]
        explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

        fig1, ax1 = plt.subplots()
        fig1.suptitle('Sample Distribution', fontsize=14, fontweight='bold')
        ax1.text(0, 0, 'Configuration\nsampleLength : {}s\nmaxPauseLength  : {}s\ntotal number of samples : {}'.format(self.sampleLength, self.maxPauseLength, numPositives + numNegatives),
                 style='italic',
                 bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        # Equal aspect ratio ensures that pie is drawn as a circle.
        ax1.axis('equal')
        if saveTo:
            plt.savefig(saveTo)
        else:
            plt.show()

    @timeit
    def grapFromVideo(self, path=None, numSamples=100, **kwargs):
        """
        only to compare the time needed to grap samples from videos to the time needed to load samples from disk.
        """
        samples = []
        for sample in self.getAllSamples("faceImage", path, relative=True, **kwargs):
            samples.append(sample)
            if len(samples) == numSamples:
                break
        return samples

    @timeit
    def grapFromDisk(self, sampleFolder, **kwargs):
        """
        only to compare the time needed to grap samples from videos to the time needed to load samples from disk.
        """
        samples = []
        files = glob.glob(os.path.join(sampleFolder, "*.pickle"))
        for file in files:
            s = FeatureizedSample()
            s.load(file)
            samples.append(s)
        return samples


def saveBalancedDataset(dataset, saveTo, featureType, shape, path=None, ratioPositives=2, ratioNegatives=1, showStatus=False, **kwargs):
    """
    saves a balanced dataset to disk
    """
    positivesFolder = os.path.join(saveTo, "positiveSamples")
    negativesFolder = os.path.join(saveTo, "negativeSamples")
    if not os.path.exists(positivesFolder):
        os.makedirs(positivesFolder)
    if not os.path.exists(negativesFolder):
        os.makedirs(negativesFolder)

    if showStatus:
        ts = time.perf_counter()
        dataset.debugPrint("[getAllPSamples] ###### Status:   0% done")
    if not path:
        path = dataset.path
    folders = list(os.walk(path, followlinks=True))[0][1]
    folders.sort()
    # construct params for producer
    params = []
    for folder in folders:
        currentFolder = os.path.abspath(os.path.join(path, folder))
        # pool.apply_async(producer, producerParams)# #Callback could also be applied
        # producer(dataset, [currentFolder, featureType, shape])
        pool.apply_async(
            producer, (dataset, [currentFolder, featureType, shape]))
        # Why does it not start????????
    # start consumer in Thread
    p = multiprocessing.Process(target=consumer, args=(
        positivesFolder, negativesFolder, ratioPositives, ratioNegatives))
    p.start()
    # Pool.join()
    pool.close()
    pool.join()
    # kill consumer
    p.terminate()

    self.debugPrint(
        "[saveBalancedDataset] Saved balanced dataset! {} samples were droped.".format(dataset.dropouts))


def transformToHDF5(path, hdf5_path, validation_split=0.2, testing=False):
    """
    transform a pickled dataset to one big hdf5 file.

    :param path: path to the folder containing the folders positiveSamples and negativeSamples
    :type path: String
    :param hdf5_path: folder to where we want to save the hdf5 files
    :type hdf5_path: String
    """
    if not os.path.exists(hdf5_path):
        print('[INFO]: path does not exist - create it')
        os.makedirs(hdf5_path)

    allPickles = glob.glob(path + '/**/*.pickle', recursive=True)
    if not testing:
        assert len(allPickles) == 22245 + \
            44489, "You didnt get alle the samples - make sure the path is correct!"

    np.random.shuffle(allPickles)
    validationPickles = allPickles[:int(len(allPickles) * validation_split)]
    trainPickles = allPickles[int(len(allPickles) * validation_split):]

    s = FeatureizedSample()
    s.load(allPickles[0])
    train_x_shape = (len(trainPickles), *s.getData().shape)
    train_y_shape = (len(trainPickles), )
    valid_x_shape = (len(validationPickles), *s.getData().shape)
    valid_y_shape = (len(validationPickles), )
    x_dtype = s.getData().dtype
    y_dtype = np.uint8

    # train
    with h5py.File(os.path.join(hdf5_path, 'vvad_train.hdf5'), mode='w') as hdf5_file:
        hdf5_file.create_dataset('X', shape=train_x_shape, dtype=x_dtype)
        hdf5_file.create_dataset('Y', shape=train_y_shape, dtype=y_dtype)

        for i, sample in enumerate(trainPickles):
            pr = (i / len(trainPickles))*100
            print('\r', 'Writing training data: {:.2f}%\r'.format(pr), end='')
            s = FeatureizedSample()
            s.load(sample)
            x = s.getData()
            y = s.getLabel()
            hdf5_file['X'][i] = x
            hdf5_file['Y'][i] = y
        print('\r', 'Writing training data: {:.2f}%\r'.format(100.0), end='')
        print()

    # validation
    with h5py.File(os.path.join(hdf5_path, 'vvad_validation.hdf5'), mode='w') as hdf5_file:
        hdf5_file.create_dataset('X', shape=valid_x_shape, dtype=x_dtype)
        hdf5_file.create_dataset('Y', shape=valid_y_shape, dtype=y_dtype)

        for i, sample in enumerate(validationPickles):
            pr = (i / len(validationPickles))*100
            print(
                '\r', 'Writing validation data: {:.2f}%\r'.format(pr), end='')
            s = FeatureizedSample()
            s.load(sample)
            x = s.getData()
            y = s.getLabel()
            hdf5_file['X'][i] = x
            hdf5_file['Y'][i] = y
        print('\r', 'Writing validation data: {:.2f}%\r'.format(100.0), end='')
        print()


def transformPointsToNumpy(points):
    array = []
    for point in points:
        array.append([point.x, point.y])
    return np.array(array)


def transformToFeatures(path, shapeModelPath=None, shape=None):
    """
    get a Sample of type faceImage and transforms to lipImage, faceFeatures and lipFeatures
    """
    ffg = FaceFeatureGenerator(
        "allwfaceImage", shapeModelPath=shapeModelPath, shape=shape)
    input_sample = FeatureizedSample()
    input_sample.load(path)
    # # get all settings
    # numSteps = input_sample.getData().shape[0]
    # origImageSize = input_sample.getData().shape[1:]
    # lipImage, faceFeatures, lipFeatures = ffg.getFeatures(input_sample.getData()[0])
    # faceFeatures = transformPointsToNumpy(faceFeatures)
    # lipFeatures = transformPointsToNumpy(lipFeatures)
    # feature_dtype = faceFeatures.dtype
    # print(numSteps)
    # print(feature_dtype)
    # print(origImageSize)

    lipImages = []
    faceFeaturesList = []
    lipFeaturesList = []
    # lipImages = np.empty(, dtype=np.uint8)
    for i, frame in enumerate(input_sample.getData()):
        lipImage, faceFeatures, lipFeatures = ffg.getFeatures(
            input_sample.getData()[i])
        faceFeatures = transformPointsToNumpy(faceFeatures)
        lipFeatures = transformPointsToNumpy(lipFeatures)
        lipImages.append(lipImage)
        faceFeaturesList.append(faceFeatures)
        lipFeaturesList.append(lipFeatures)

    lipImageSample = FeatureizedSample()
    lipImageSample.data = np.array(lipImages)
    lipImageSample.k = len(lipImages)
    lipImageSample.label = input_sample.getLabel()
    lipImageSample.featureType = "lipImage"

    faceFeaturesSample = FeatureizedSample()
    faceFeaturesSample.data = np.array(faceFeaturesList)
    faceFeaturesSample.k = len(faceFeaturesList)
    faceFeaturesSample.label = input_sample.getLabel()
    faceFeaturesSample.featureType = "faceFeatures"

    lipFeaturesSample = FeatureizedSample()
    lipFeaturesSample.data = np.array(lipFeaturesList)
    lipFeaturesSample.k = len(lipFeaturesList)
    lipFeaturesSample.label = input_sample.getLabel()
    lipFeaturesSample.featureType = "lipFeatures"

    # save the samples in a folder next to the original dataset
    # TODO: extract base path
    path = Path(path)
    # getFilename
    fileName = os.path.join(PurePath(path.parent).name, PurePath(path).name)

    # path for new dataset
    # Assuming the folders positiveSamples and negativeSamples exist
    topFolder = path.parent.parent.parent
    lipImageFolder = os.path.join(topFolder, 'lipImageDataset')
    faceFeaturesFolder = os.path.join(topFolder, 'faceFeaturesDataset')
    lipFeaturesFolder = os.path.join(topFolder, 'lipFeaturesDataset')
    posNeg = ['positiveSamples', 'negativeSamples']
    dataFolders = [lipImageFolder, faceFeaturesFolder, lipFeaturesFolder]

    for folder in dataFolders:
        for subFolder in posNeg:
            try:
                os.makedirs(os.path.join(folder, subFolder))
            except FileExistsError:
                pass

    lipImageSample.save(os.path.join(lipImageFolder, fileName))
    faceFeaturesSample.save(os.path.join(faceFeaturesFolder, fileName))
    lipFeaturesSample.save(os.path.join(lipFeaturesFolder, fileName))

    # TODO:call this in a multiproccessing way.


def makeTestSet(path, namesPath):
    """
    takes the names belonging to the testset from the dataset in path

    :param path: Path to the dataset with positiveSamples and negativeSamples folder
    :type path: String
    :param namesPath: pickleFile with a list of all the fileNames belonging to the testset
    :type namesPath: String
    """
    testSetPath = os.path.join(path, 'testSet')
    testPos = os.path.join(testSetPath, 'positiveSamples')
    if not os.path.exists(testPos):
        os.makedirs(testPos)
    testNeg = os.path.join(testSetPath, 'negativeSamples')
    if not os.path.exists(testNeg):
        os.makedirs(testNeg)

    with open(namesPath, 'rb') as namesFile:
        names = pickle.load(namesFile)

    for name in names:
        print('NAME: {}'.format(name))
        namePath = os.path.join(path, name + ".pickle")
        if os.path.exists(namePath):
            os.rename(namePath, os.path.join(testSetPath, name + '.pickle'))
        else:
            print('NAMEPATH not existing: {}'.format(namePath))
        # if len(namePath) == 2:
        #     for sample in namePath:
        #         s = FeatureizedSample()
        #         s.load(sample)
        #         print("FEATURETYPE: {}".format(s.featureType))
        #         print("DTYPE: {}".format(s.getData().dtype))
        #         s.visualize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help='path to the config file', type=str)
    parser.add_argument("option", help="what you want to do.", type=str, choices=[
                        "makeTestSet", "tests", "analyze", "convertAll", "pSamples", "pSamplesAll", "samples", "download"])  # TODO: remove unused
    parser.add_argument("-d", "--dataPath",
                        help="the path to the dataset", type=str)
    parser.add_argument("-m", "--shapeModelPath",
                        help="the path of folder containing the models", type=str)
    parser.add_argument(
        "--debug", help="debug messages will be displayed", action='store_true')
    parser.add_argument("-p", "--maxPauseLength",
                        help="defines when to break a positive sample in seconds", type=float)
    parser.add_argument("-l", "--sampleLength",
                        help="defines the minimum length of a sample in seconds", type=float)
    parser.add_argument(
        "-f", "--fps", help="the frames per second on the videos", type=int)
    parser.add_argument(
        "-s", "--shape", help="[x,y] defines the size to which face or lip images will be resized - this is the input size of the net", type=list)
    parser.add_argument(
        "-n", "--names", help="path to the names pickle file", type=str)

    args = parser.parse_args()

    # get values from config
    config = yaml.load(open(args.config))
    ### Config Values ###
    dataPath = args.dataPath if args.dataPath else config["dataPath"]
    shapeModelPath = args.shapeModelPath if args.shapeModelPath else config["shapeModelPath"]
    debug = args.debug if args.debug else config["debug"]
    maxPauseLength = args.maxPauseLength if args.maxPauseLength else config["maxPauseLength"]
    sampleLength = args.sampleLength if args.sampleLength else config["sampleLength"]
    shape = args.shape if args.shape else config["shape"]
    fps = args.fps if args.fps else config["fps"]

    ds = DataSet(shapeModelPath, debug=debug, sampleLength=sampleLength,
                 maxPauseLength=maxPauseLength, shape=shape, path=dataPath, fps=fps)
    if args.option == "download":
        ds.downloadLRS3(dataPath)
    if args.option == "pSamples":
        for sample in ds.getPositiveSamples(dataPath):
            sample.visualize("mouthImages")
    if args.option == "pSamplesAll":
        for sample in ds.getAllPSamples(dataPath):
            pass
    if args.option == "convertAll":
        ds.convertAllFPS(dataPath)
    if args.option == "analyze":
        pauses = ds.analyzeNegatives(dataPath)
        pSamples = ds.analyzePositives(dataPath)
    if args.option == "tests":
        # samples = ds.grapFromVideo(numSamples=100, dryRun=True)
        # for i, sample in enumerate(samples):
        #     print("Sample number {}".format(i))
        #     assert len(sample) == 2, "SampleConfig needs to have a len of 2 has {}\nsampleConfig: {}".format(len(sample), sample)
        #     assert len(sample[1]) == 6, "inner SampleConfig needs to have a len of 6 has {}\ninner sampleConfig: {}".format(len(sample[1]), sample[1])
        #     print(sample)
        ds.analyzeNegatives()

        def testTime():
            logtime_data = {}
            testFolder = "TestPickles"
            if not os.path.exists(testFolder):
                os.makedirs(testFolder)
            samples = ds.grapFromVideo(numSamples=100, log_time=logtime_data)
            for c, sample in enumerate(samples):
                sample.save(os.path.join(testFolder, str(c) + ".pickle"))
                print("saved sample {}".format(c))

            samples = ds.grapFromDisk(testFolder, log_time=logtime_data)

            print(logtime_data)
        # testTime()
        #saveBalancedDataset(ds, "../data/balancedCleandDataSet/", "faceImage", shape, showStatus=True)
        # samples = ds.grapFromDisk("TestPickles")
        # for sample in samples:
        #     if not sample.label:
        #         sample.visualize()

        # ds.analyze(saveTo="../thesis/HgbThesisEN/images/sampleDistribution.png")

        # c = 0
        # for sample in ds.getAllSamples(featureType = "faceImage", relative = True):
        #     c += 1
        #     print("Sample {}".format(c))
        #     sample.save("testSample.pickle")
        #     break
        # loadedSample = FeatureizedSample()
        # loadedSample.load("testSample.pickle")
        # loadedSample.visualize()

    if args.option == "makeTestSet":
        makeTestSet(args.dataPath, args.names)
