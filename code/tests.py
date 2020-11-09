#system imports
#from imutils import paths

# 3rd party imports
#import cv2

# local imports
# from imageUtils import *
# from videoUtils import *
from dataSet import *
import multiprocessing
import time
import pickle

import numpy as np
import matplotlib.pyplot as plt
#
if __name__ == "__main__":


    mu, sigma = 100, 15
    x = mu + sigma * np.random.randn(10000)
    n, bins, patches = plt.hist(x, 50, normed=1, facecolor='g', alpha=0.75)
    plt.xlabel('Smarts', size = 20)
    plt.ylabel('Probability')
    # plt.title('Histogram of IQ', size = 50)
    # plt.text(60, .025, r'$\mu=100,\ \sigma=15$', size = 30)
    plt.axis([40, 160, 0, 0.03])
    plt.xticks(size = 30)
    plt.yticks(size = 30)
    plt.grid(True)
    plt.show()

#
#     parser = argparse.ArgumentParser()
#     parser.add_argument("config", help='path to the config file' , type=str)
#     parser.add_argument("option", help="what you want to do.", type=str, choices=["tests", "analyze","convertAll", "pSamples","pSamplesAll", "samples", "download"])#TODO: remove unused
#     parser.add_argument("-d","--dataPath", help="the path to the dataset" , type=str)
#     parser.add_argument("-m", "--shapeModelPath",help="the path of folder containing the models" , type=str)
#     parser.add_argument("--debug",help="debug messages will be displayed", action='store_true')
#     parser.add_argument("-p", "--maxPauseLength",help="defines when to break a positive sample in seconds" , type=float)
#     parser.add_argument("-l", "--sampleLength",help="defines the minimum length of a sample in seconds", type=float)
#     parser.add_argument("-f", "--fps",help="the frames per second on the videos", type=int)
#     parser.add_argument("-s", "--shape",help="[x,y] defines the size to which face or lip images will be resized - this is the input size of the net", type=list)
#
#     args = parser.parse_args()
#
#     # get values from config
#     config = yaml.load(open(args.config))
#     ### Config Values ###
#     dataPath = args.dataPath if args.dataPath else config["dataPath"]
#     shapeModelPath = args.shapeModelPath if args.shapeModelPath else config["shapeModelPath"]
#     debug = args.debug if args.debug else config["debug"]
#     maxPauseLength = args.maxPauseLength if args.maxPauseLength else config["maxPauseLength"]
#     sampleLength = args.sampleLength if args.sampleLength else config["sampleLength"]
#     shape = args.shape if args.shape else config["shape"]
#     fps = args.fps if args.fps else config["fps"]
#
#
#
#     ds = DataSet(shapeModelPath, debug=debug, sampleLength = sampleLength, maxPauseLength=maxPauseLength, shape=shape, path=dataPath, fps = fps)
#     with open("Test.pickle", 'wb') as file:
#         pickle.dump(ds, file)
#
#     def murks(ds):
#         ds.debugPrint("Test")
#
#     pool = multiprocessing.Pool()
#     for x in range(100):
#         pool.apply_async(murks, (ds,))
#     pool.close()
#     pool.join()
#
#     # a = FeatureizedSample()
#     # a.load("testSample.pickle")
#     # a.visualize()
#
#     # parser = argparse.ArgumentParser()
#     # parser.add_argument("videoPath", help='path to the video file' , type=str)
#     # parser.add_argument("modelPath", help='path to the model file' , type=str)
#     # parser.add_argument("featureType", help='type of the features' , type=str)
#     #
#     # args = parser.parse_args()
# # Test FeatureizedSample
#
#
#
#
#
#
# # #TESTVIDEO
# # ffg = FaceFeatureGenerator(args.featureType, modelPath = args.modelPath, shape = (200,200))
# # firstImage = True
# # for success, image in getFramesfromVideo(args.videoPath):
# #     if success:
# #         if firstImage:
# #             # cv2.imshow("Image", image)
# #             # cv2.waitKey(0)
# #             #getRandomFaceBox
# #             faceBox = getRandomFaceFromImage(image)
# #             if faceBox:
# #                 firstImage = False
# #                 tracker = FaceTracker(faceBox)
# #             else:
# #                 print("NO FACE")
# #         else:
# #             face, boundingBox = tracker.getNextFace(image)
# #             features = ffg.getFeatures(face)
# #             if "Image" in args.featureType :
# #                 cv2.imshow(args.featureType, features)
# #                 key = cv2.waitKey(1) & 0xFF
# #                 # if the `q` key was pressed, break from the loop
# #                 if key == ord("q"):
# #                     break
#
#
# #TestFaceTracker
#
#
#
#
#
# #TestResizeAndZeroPadding
# # imagePaths = sorted(list(paths.list_images("/home/al/Downloads")))
# #
# # for imagePath in imagePaths:
# #     image = cv2.imread(imagePath)
# #     print(imagePath)
# #     image = resizeAndZeroPadding(image, (200, 200))
# #     # cv2.imshow("", image)
# #     # cv2.waitKey(0)
