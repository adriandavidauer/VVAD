"""
This Module creates a dataset for the purpose of the visual speech detection system.
"""
# System imports
import os
import pathlib
# System imports
import sys
from importlib import import_module
# from collections import deque
import pathlib
from pathlib import Path, PurePath

import dlib
import h5py
import yaml
from ffmpy import FFmpeg
from file_read_backwards import FileReadBackwards
# 3rd party imports
from pytube import YouTube

from vvadlrs3.sample import *
import h5py
import yaml
from file_read_backwards import FileReadBackwards
# 3rd party imports
from pytube import YouTube

# local imports
# from vvadlrs3.utils.multiprocessingUtils import *
from vvadlrs3.sample import *
from vvadlrs3.utils.multiprocessingUtils import *
from vvadlrs3.utils.timeUtils import *

# end file header
__author__ = "Adrian Lubitz"
__copyright__ = "Copyright (c)2017, Blackout Technologies"


class WrongPathException(Exception):
    "Data folder is probably not mounted. Or you gave the wrong path."
    # print("Data folder is probably not mounted. Or you gave the wrong path.")
    pass


class DataSet:
    """
    This class handles everything involved with the datasets.
    From creation and downloading over cleaning and balancing to converting and
    displaying.
    """

    # TODO: add path to Parameters
    def __init__(self, shape_model_path, debug_flag, sample_length, max_pause_length,
                 init_shape, path, target_fps,
                 init_multiprocessing):
        """
        Just initializing an empty dataset
        """
        self.tempPath = None
        self.shapeModelPath = shape_model_path
        self.debug = debug_flag
        self.sampleLength = sample_length
        self.maxPauseLength = max_pause_length
        self.k = int(round(self.get_frame_from_second(self.sampleLength)))
        self.path = path
        self.fps = target_fps
        self.shape = init_shape
        self.dropouts = 0
        self.multiprocessing = False

        if init_multiprocessing:
            self.multiprocessing = True
            self.importedModule = import_module('vvadlrs3.utils.multiprocessingUtils')

    def debug_print(self, debug_msg):
        """
        printing debug message if debug is set.
        """
        if self.debug:
            print(debug_msg)

    def download_lrs3_sample_from_youtube(self, path):
        """
        downloading corresponding video data for the LRS3 dataset from youtube

        Args:
            path (str): Path to a folder containing the txt files
        """

        current_folder = os.path.abspath(path)
        # print (currentFolder)
        # open folder and get a list of files
        try:
            files = list(os.walk(current_folder, followlinks=True))[0][2]
        except:
            raise WrongPathException
        files = [pathlib.Path(os.path.join(current_folder, file))
                 for file in files]
        # get the RefField
        for file in files:
            if file.suffix == ".txt":
                text_file = open(file)
                # hat anscheinend noch ein return mit drinne
                ref = text_file.readlines()[2][7:].rstrip()
                break

        # Prep ref checking
        video_file_without_extension = pathlib.Path(
            os.path.join(current_folder, ref))
        # check if video is already there
        already_downloaded = False
        for file in files:
            if file.suffix != ".txt":  # A video is there
                if ref in file.resolve().stem:  # Fully downloaded
                    print("Video already downloaded")
                    already_downloaded = True
                else:
                    print("Restarting download of unfinished video")
                    os.remove(file)
                break
        if not already_downloaded:
            video_url = "https://www.youtube.com/watch?v={}".format(ref)
            print("starting to download video from {}".format(video_url))
            # download in a temp file (will be the title of the Video in youtube)
            self.tempPath = None

            global timeoutable_download

            def timeoutable_download(url_video, folder_current):
                self.tempPath = YouTube(url_video).streams.first().download(
                    folder_current)
                self.tempPath = pathlib.Path(self.tempPath)
                # if ready rename the file to the real name(will be the ref)
                os.rename(self.tempPath, str(
                    video_file_without_extension) + self.tempPath.resolve().suffix)

            if self.multiprocessing:
                p = Process(target=timeoutable_download,
                            args=(video_url, current_folder))

                print("name is: ", __name__)
                if __name__ == '__main__':
                    multiprocessing.freeze_support()
                    print("in if")
                    p.start()
                    p.join(600)
                    if p.is_alive():
                        print("Timeout for Download reached!")
                        # Terminate
                        p.terminate()
                        p.join()
            else:
                timeoutable_download(video_url, current_folder)

    # TODO add option if you want to use whats there or download if neccessary
    def get_all_p_samples(self, path, **kwargs):
        """
        making all the samples from this folder.

        Args:
            path (str): Path to the DataSet folder containing folders, which
            contain txt files. (For Example the pretrain folder)
        """
        print("my path", path)
        folders = list(os.walk(path, followlinks=True))[0][1]
        print("folders are: ", folders)
        folders.sort()
        for folder in folders:
            # Video file is the only not txt file
            currentFolder = os.path.abspath(os.path.join(path, folder))
            for current_sample in self.get_positive_samples(currentFolder, **kwargs):
                yield current_sample
            self.debug_print("[getAllPSamples] Folder {} done".format(folder))

    # TODO add option if you want to use whats there or download if necessary
    def get_all_samples(self, feature_type, path=None, relative=True, dry_run=False,
                        showStatus=False, **kwargs):
        """
        making all the samples from this folder.

        Args:
            path (str): Path to the DataSet folder containing folders, which contain
            txt files. (For Example the pretrain folder)
        """
        if showStatus:
            ts = time.perf_counter()
            self.debug_print("[getAllPSamples] ###### Status:   0% done")
        if not path:
            path = self.path
        folders = list(os.walk(path, followlinks=True))[0][1]
        folders.sort()
        for i, folder in enumerate(folders):
            # Video file is the only not txt file
            current_folder = os.path.abspath(os.path.join(path, folder))
            for single_sample in self.get_samples(current_folder,
                                                  feature_type=feature_type,
                                                  # relative=relative,
                                                  samples_shape=(200, 200),
                                                  dry_run=dry_run):
                yield single_sample
            self.debug_print("[getAllPSamples] Folder {} done".format(folder))
            if showStatus:
                self.debug_print("[getAllPSamples] ###### Status:   {}% done".format(
                    float(i) / len(folders) * 100))
                self.debug_print("[getAllPSamples] ### Time elapsed: {} ms".format(
                    (time.perf_counter() - ts) * 1000))

    def convert_all_fps(self, path):
        """
        converting all the fps from this folder.

        Args:
        path (str): Path to the DataSet folder containing folders, which contain txt
        files. (For Example the pretrain folder)
        """
        folders = list(os.walk(path, followlinks=True))[0][1]
        folders.sort()
        for folder in folders:
            # Video file is the only not txt file
            current_folder = os.path.abspath(os.path.join(path, folder))
            try:
                self.convert_fps(current_folder)
            except FileNotFoundError as e:
                self.debug_print(str(e) + "Skipping folder")

    def download_lrs3(self, path):
        """
        downloading corresponding video data for the LRS3 dataset from youtube and
        saving the faceFrames in the corrosponding folder

        Args
            path (str): Path to the DataSet folder containing folders, which contain
            txt files. (For Example the pretrain folder)
        """

        # for folder in path call cutTedVideo - need to extract the Video File first
        folders = list(os.walk(path, followlinks=True))[0][1]
        folders.sort()
        for folder in folders:
            # Video file is the only not txt file
            current_folder = os.path.abspath(os.path.join(path, folder))
            self.download_lrs3_sample_from_youtube(current_folder)

    @staticmethod
    def get_txt_files(path):
        """
        Get all the txt files with a generator from the path

        Args:
            path (str): Path to txt files from a video
        """
        try:
            files = list(os.walk(current_folder, followlinks=True))[0][2]
        except WrongPathException:
            raise WrongPathException
        files = [pathlib.Path(os.path.join(current_folder, file))
                 for file in files]
        for file in files:
            if file.suffix == ".txt":
                yield file

    def get_positive_samples(self, path, dry_run=False):
        """
        Returning all positive samples from a Video with a generator

        Args:
            path (str): Path to a folder containing the txt files
            dryRun (bool): With a dry run you will not really return samples, just a
            list of tuples with start and end time of the positive samples

        Returns:
            generator
        """
        try:
            videoPath = self.getVideoPathFromFolder(path)
        except FileNotFoundError as e:
            self.debug_print(e)
            return []  # No Samples...sorry :/

        # self.debugPrint(videoPath)

        folder = os.path.dirname(video_path)
        # list of configs [startFrame, endFrame , x, y, w, h] x,y,w,h are rel. pixels
        frame_list = []
        print("framelist: ", frame_list)
        # for every txt file
        for textFile in self.get_txt_files(folder):
            frame_list.extend(self.get_sample_configs_for_pos_samples(textFile))
            # firstFrameLine = ""
            # lastFrameLine = ""
            # textFile = open(textFile)
            # for line in textFile.readlines()[5:]:
            #     line = line.rstrip()
            #     if not firstFrameLine:# only the first line of frames will be saved
            #         firstFrameLine = line
            #     # if line is empty - last line of the frames
            #     if not line:
            #         break
            #     lastFrameLine = line
            # firstFrame = firstFrameLine.split()
            # lastFrame = lastFrameLine.split()
            #
            # configList = [int(firstFrame[0]), int(lastFrame[0]), float(firstFrame[1]),
            #   float(firstFrame[2]),
            #   float(firstFrame[3]), float(firstFrame[4])]
            # frameList.append(configList)

        frame_list.sort(key=lambda x: x[0])

        # Open video
        if not dry_run:
            video_path = self.convert_fps(video_path.parents[0])
            vid_obj = cv2.VideoCapture(str(video_path))
            vid_fps = vid_obj.get(cv2.CAP_PROP_FPS)
        count = 0
        # sampleList = []
        for sampleConfig in frame_list:
            if not self.check_sample_length(
                    self.get_second_from_frame(sampleConfig[0]),
                    self.get_second_from_frame(sampleConfig[1])):
                continue
            if not dry_run:
                data = []
                label = True
                config = {"x": sampleConfig[2], "y": sampleConfig[3],
                          "w": sampleConfig[4], "h": sampleConfig[5], "fps": vid_fps}
                # grap frames from start to endframe
                while True:
                    success, image = vid_obj.read()
                    if not success:
                        raise Exception(
                            "Couldnt grap frame of file {}".format(video_path))
                    if sampleConfig[0] <= count <= sampleConfig[1]:
                        data.append(image)
                    count += 1
                    if count > sampleConfig[1]:
                        break
                yield Sample(data, label, config, self.shapeModelPath)
            else:
                yield self.get_second_from_frame(sampleConfig[0]), \
                      self.get_second_from_frame(sampleConfig[1])

    def convert_fps(self, path, fps=25):
        """
        converting video in path to fps

        Args:
            path (str): Path to a folder containing the txt files
            fps (float): frames per second
        """
        video_path = self.get_video_path_from_folder(path)

        vid_obj = cv2.VideoCapture(str(video_path))
        vid_fps = vid_obj.get(cv2.CAP_PROP_FPS)
        if vid_fps != fps:
            # change the frameRate to 25, because the data set is expecting that!
            # ffmpeg -y -r 30 -i seeing_noaudio.mp4 -r 24 seeing.mp4
            old_video_path = video_path
            video_path = pathlib.Path(os.path.join(
                old_video_path.parents[0], old_video_path.stem + ".converted" +
                                           old_video_path.suffix))

            command = f"ffmpeg -i {old_video_path} -filter:v fps:{fps} {video_path}"

            print(command)
            os.system(command)

            # print(changeFps.cmd)
            # stdout, stderr = change_fps.run()
            # Remove the old!
            #ToDo Remove old video path
            #os.remove(old_video_path)
            self.debug_print("Changed FPS of {} to {}".format(video_path, fps))
        else:
            self.debug_print("{} has already the correct fps".format(video_path))
        return video_path

    def get_video_path_from_folder(self, path):
        """
        Get the path to the only video in folder raises Exception if there is none and
        removes all invalid files.

        Args:
            path (str): path to video folder
        """
        current_folder = os.path.abspath(path)
        video_path = None
        try:
            files = list(os.walk(current_folder, followlinks=True))[0][2]
        except IndexError:
            raise WrongPathException
        files = [pathlib.Path(os.path.join(current_folder, file))
                 for file in files]
        video_files = []
        for file in files:
            if file.suffix not in [".txt"]:  # A video is there
                video_files.append(file)
        if not video_files:
            raise FileNotFoundError("No video in {}".format(current_folder))
        if len(video_files) > 1:
            self.debug_print(
                "TOO MANY VIDEOS OR OTHER FILES IN {}".format(current_folder))
            for video in video_files:
                if ".converted" in video.stem:  # if there are two videos, don't remove
                    # the original!
                    self.debug_print("Deleting {}".format(video))
                    os.remove(video)
                    return self.get_video_path_from_folder(current_folder)
        else:
            video_path = video_files[0]

        return pathlib.Path(os.path.abspath(video_path))

    def analyze_negatives(self, path=None, save_to=None):
        """
        Showing/Saving statistics over the data set.

        Args:
            path (str): Path to the DataSet folder containing folders, which contain
            txt files. (For Example the pretrain folder)
        """
        if not path:
            path = self.path
        folders = list(os.walk(path, followlinks=True))[0][1]
        folders.sort()
        num_total_samples = 0
        pauses = []
        for folder in folders:
            current_folder = os.path.abspath(os.path.join(path, folder))
            # for every txt file
            for textFile in self.get_txt_files(current_folder):
                num_total_samples += 1
                video_pauses = self.get_pause_length(textFile)
                # if videoPauses:#
                #     print("VideoPauses: {}".format(videoPauses))
                pauses.extend(video_pauses)
            self.debug_print("[analyzeNegatives] Folder {} done".format(folder))
        # TODO: norm to the number of analyzedSamples to see how many negative Samples
        #  can be constructed out of how many positive samples

        hist_data = [x[1] - x[0]
                     for x in pauses if self.check_sample_length(x[0], x[1])]
        self.debug_print(
            "Number of extracted negative samples:  {}".format(len(hist_data)))
        # bins=15)#np.arange(1.0, 19.0))
        plt.hist(hist_data, np.arange(self.sampleLength, max(hist_data)))
        plt.ylabel('Num Negatives', size=30)
        plt.xlabel('Sample Length', size=30)
        plt.xticks(size=30)
        plt.yticks(size=30)
        print("Total Amount of Samples is {}".format(num_total_samples))
        if save_to:
            plt.savefig(save_to)
        else:
            plt.show()
        # return sorted(pauses, key = lambda x: x[0] - x[1])
        return pauses

    def analyze_positives(self, path, save_to=None):
        """
        Showing/Saving statistics over the data set.

        Args:
            path (str): Path to the DataSet folder containing folders, which contain
            txt files. (For Example the pretrain folder)
        """
        p_samples = self.get_all_p_samples(path, dry_run=True)
        # TODO: norm to the number of analyzedSamples to see how many negative Samples
        #  can be constructed out of how many positive samples

        hist_data = [x[1] - x[0]
                     for x in p_samples if self.check_sample_length(x[0], x[1])]
        self.debug_print(
            "Number of extracted positive samples:  {}".format(len(hist_data)))
        plt.hist(hist_data, np.arange(self.sampleLength, max(hist_data)))
        plt.ylabel('Num positive Samples')
        plt.xlabel('Sample Length')
        if save_to:
            plt.savefig(save_to)
        else:
            plt.show()
        # return sorted(pauses, key = lambda x: x[0] - x[1])
        return p_samples, len(hist_data)

    @staticmethod
    def get_frame_from_second(second, fps=25):
        """
        calculates the frame in a video from a given second. (rounded off)

        Args:
            second (float): second in the video
            fps (float): frame rate of video in frames per second

        Returns:
            frame (float): frame in video as float, rounding needs to be made explicit
        """
        return float(second * fps)

    @staticmethod
    def get_second_from_frame(frame, input_fps=25):
        """
        calculates the second in a video from a given frame.

        Args:
            frame (float): frame in the video
            fps (float): framerate of video

        Returns:
            time (float): Time of frame in video (seconds)
        """
        return float(frame) / input_fps

    def get_pause_length(self, txt_file):
        """
        returns the length auf pauses and corrosponding start and end frame.

        Args:
            txtFile (str): Path to the txt file

        Returns:
            pauses (list): List of pauses in the txt meta data of a video
        """
        last_start = None
        pauses_list = []  # list of pauses defined by a tuple (startTime, endTime)
        with FileReadBackwards(txt_file) as txt:
            for e in txt:
                if "WORD START END ASDSCORE" in e:
                    break
                word, start, end, asdscore = e.split()
                # self.debugPrint("WORD: {} START: {} END: {} ASDSCORE: {}".format(word,
                # start, end, asdscore))
                # end < lastStart:
                if last_start and \
                        (float(last_start) - float(end) > self.maxPauseLength):
                    # there is a pause
                    # if float(lastStart) - float(end) > 15.0:
                    #     print("Check out sample from {} where word is {}".format(
                    #     txtFile, word))
                    pauses_list.append((float(end), float(last_start)))
                    # self.debugPrint(pauses)
                last_start = start
        # return sorted(pauses, key = lambda x: x[0] - x[1])
        return pauses_list

    def get_sample_configs_for_pos_samples(self, txt_file):
        """
        returns a list of Frame configs for positive samples
        [startFrame, endFrame , x, y, w, h] x,y,w,h are relative pixels

        Args:
            txtFile (str): Path to the txt file

        Returns:
            Frame configs (list): positive samples [startFrame, endFrame , x, y, w, h]
        """

        # check for Pauses
        pauses = self.getPauseLength(txtFile)
        # translate to Frames
        pauses = sorted([(int(np.ceil(self.getFrameFromSecond(x[0]))), int(
            self.getFrameFromSecond(x[1]))) for x in pauses], key=lambda x: x[0])

        # for all frames make a sample from start to pause0_start and from pause0_end
        # to pause1_start ... pauseN_end to end
        first_frame_config = []
        last_frame_config = []
        pause_start = []
        pause_end = []
        text_file = open(txt_file)
        config_list = []
        for line in text_file.readlines()[5:]:
            current_config = line.rstrip().split()
            if not first_frame_config:  # only the first line of frames will be saved
                first_frame_config = current_config
                # add first frame num of the sample to all values of the pauses,
                # because pauses are relative to start
                pauses = [(x[0] + int(first_frame_config[0]), x[1] +
                           int(first_frame_config[0])) for x in pauses]
            # check if the currentFrame is a pauseStart or pauseEnd frame
            if pauses:
                # self.debugPrint("currentConfig: {}\npauses: {}\nSample: {}".format(
                # currentConfig, pauses, txtFile))
                if int(current_config[0]) in pauses[0]:
                    if not pause_start and not pause_end:
                        # first sample
                        pause_start = current_config
                        # from start to pauseStart
                        config_list.append(
                            [int(first_frame_config[0]), int(pause_start[0]),
                             float(first_frame_config[1]), float(
                                first_frame_config[2]), float(first_frame_config[3]),
                             float(first_frame_config[4])])

                    elif not pause_end and pause_start:
                        # pauseStart is set so what I get is pauseEnd - just empty
                        # pauseStart and pop from pauses
                        pause_start = []
                        pauses.pop(0)
                    elif not pause_start and pause_end:
                        # pauseEnd is set so what I get is pauseStart - from pauseEnd
                        # to pauseStart and empty pauseEnd
                        config_list.append([int(pause_end[0]), int(pause_start[0]),
                                            float(pause_end[1]), float(pause_end[2]),
                                            float(pause_end[3]), float(pause_end[4])])
                        pause_end = []
                    else:
                        # shouldnt happen!!
                        raise Exception("WTF")
            # if line is empty - last line of the frames
            if not line.rstrip():
                # put last sample
                # from pauseEnd to lastFrameConfig
                if pauses:
                    config_list.append([int(pause_end[0]), int(last_frame_config[0]),
                                        float(pause_end[1]), float(pause_end[2]),
                                        float(pause_end[3]), float(pause_end[4])])
                break
            last_frame_config = current_config

        assert len(
            pauses) == 0, "pauses is not empty...should be!\npauses: {}".format(pauses)
        if not config_list:  # there where no pauses in this sample
            config_list.append([int(first_frame_config[0]), int(last_frame_config[0]),
                                float(first_frame_config[1]),
                                float(first_frame_config[2]),
                                float(first_frame_config[3]),
                                float(first_frame_config[4])])
        return config_list

    def check_sample_length(self, start, end):
        """
        returns True if end - start is bigger than self.sampleLength

        Args:
            start (float): start of the sample in seconds
            end (float): end of the sample in seconds

        Returns:
            valid (bool): If sample is long enough for data set
        """
        return (end - start) > self.sampleLength

    def get_sample_configs(self, txt_file):
        """
        returns a list of tuples holding the config of a sample consisting out of the
        following:
        [(label, [startFrame, endFrame , x, y, w, h]), ...] x,y,w,h are relative pixels
        of the bounding box in the first frame

        Args:
            txtFile (str): Path to the txt file

        Returns:
            configList (list of tuples): holding the frame config and corresponding label
        """
        pauses = self.get_pause_length(txt_file)
        # translate to Frames
        pauses = sorted([(int(np.ceil(self.get_frame_from_second(x[0]))), int(
            self.get_frame_from_second(x[1]))) for x in pauses], key=lambda x: x[0])
        text_file = open(txt_file)
        config_list = []
        negative_frames = []
        counter = False
        for line in text_file.readlines()[5:]:
            if not line.rstrip():
                break
            current_config = line.rstrip().split()
            if not counter:  # only for the first line
                # initialize counter
                counter = int(current_config[0])
                # add first frame num of the sample to all values of the pauses, because
                # pauses are relative to start
                pauses = [(x[0] + int(current_config[0]), x[1] +
                           int(current_config[0])) for x in pauses]
                # transform to list of all framenums associated with a pause/negative
                # sample
                for pause in pauses:
                    negative_frames.extend(list(range(pause[0], pause[1] + 1)))
                list_len = len(negative_frames)
                # construct a set to check for the union
                negative_frames = set(negative_frames)
                assert list_len == len(
                    negative_frames), "There are doublets in the frameList of pauses " \
                                      "that should not happen"
            assert counter >= int(current_config[0])
            # check if this frame needs to be taken in consideration
            if counter == int(current_config[0]):
                # check k frame numbers if they are in negative_frames
                k_frame_numbers = set(list(range(counter, counter + self.k)))
                intersection = negative_frames.intersection(k_frame_numbers)
                # check the three cases of the intersection
                if len(intersection) == 0:
                    # its a positive sample - save and skip k frames
                    sample_config = (True, [int(current_config[0]),
                                            int(current_config[0]) + (self.k - 1),
                                            float(current_config[1]),
                                            float(current_config[2]),
                                            float(current_config[3]),
                                            float(current_config[4])])
                    config_list.append(sample_config)
                    counter += self.k
                elif len(intersection) == self.k:
                    # its a negative sample - save and skip k frames
                    sample_config = (False, [int(current_config[0]),
                                             int(current_config[0]) + (self.k - 1),
                                             float(current_config[1]),
                                             float(current_config[2]),
                                             float(current_config[3]),
                                             float(current_config[4])])
                    config_list.append(sample_config)
                    counter += self.k
                else:
                    counter += 1  # it's a mix of negative and positive frames -> just
                    # take the next k frames
        # check if last sample is valid (could reach out of video) counter is only
        # allowed to be 1 frames bigger than currentConfig[0]
        if counter - int(current_config[0]) > 1:
            config_list.pop()
        return config_list

    def get_samples(self, path, feature_type, samples_shape, dry_run=False):
        """
        Returning all samples from a Video with a generator

        Args:
            path (str): Path to a folder containing the txt files
            dryRun (bool): With a dry run you will not really return samples, just a list of sampleConfigs

        Returns:
            generator
        """
        self.shape = samples_shape
        try:
            video_path = self.get_video_path_from_folder(path)
        except FileNotFoundError as e:
            self.debug_print(e)
            return []  # No Samples...sorry :/
        folder = os.path.dirname(video_path)
        # list of configs    [startFrame, endFrame , x, y, w, h] x,y,w,h are rel. pixels
        sample_config_list = []
        # for every txt file
        for textFile in self.get_txt_files(folder):
            sample_config_list.extend(self.get_sample_configs(textFile))
        sample_config_list.sort(key=lambda x: x[1][0])
        if not dry_run:
            video_path = self.convert_fps(video_path.parents[0])
            vid_obj = cv2.VideoCapture(str(video_path))

            frames = []
            count = 0
            success = True
            while success and sample_config_list:
                # print("Sample from Frame {} to {}".format(sampleConfigList[0][1][0],
                # sampleConfigList[0][1][1]))
                # print("Next Sample from Frame {} to {}".format(
                # sampleConfigList[1][1][0], sampleConfigList[1][1][1]))
                # print("Counter: {}".format(count))
                if len(frames) == self.k:
                    sample = FeatureizedSample()
                    sample.generate_sample_from_fixed_frames(
                        self.k, frames, sample_config_list[0][1][2:],
                        sample_config_list[0][0],
                        feature_type=feature_type,
                        shape=self.shape, shape_model_path=self.shapeModelPath)
                    if sample.is_valid():
                        yield sample
                    else:
                        # TODO: keep track of the dropouts!
                        print("invalid sample")
                        # TODO: make threadsafe!
                        #  https://docs.python.org/2/library/multiprocessing.html#
                        #  sharing-state-between-processes
                        self.dropouts += 1
                    sample_config_list.pop(0)
                    frames = []
                if sample_config_list:
                    success, image = vid_obj.read()
                    # TODO: Here is the issue that the list have been popped empty!!!!!
                    #  EdgeCase
                    if sample_config_list[0][1][0] <= count <= \
                            sample_config_list[0][1][1]:
                        frames.append(image)
                        # print ("Added Frame. len(frames): {}".format(len(frames)))
                    count += 1
        else:
            for sample in sample_config_list:
                yield sample

    def analyze(self, path=None, save_to=None):
        """
        Shows statistics over the samples(values from the config, sum samples,
        negative samples, positive samples, ...)

        Args:
            path (str): Path to samples folder
            saveTo (str): Path to save the analysis results' figure to
        """
        num_positives = 0
        num_negatives = 0
        for sampleConfig in self.get_all_samples(
                "faceImage", path, dryRun=True, showStatus=True):
            print("Sample Config is ", sampleConfig)
            if sampleConfig[0]:
                num_positives += 1
            else:
                num_negatives += 1

        labels = 'positive samples', 'negative samples'
        sizes = [num_positives, num_negatives]
        explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

        fig1, ax1 = plt.subplots()
        fig1.suptitle('Sample Distribution', fontsize=14, fontweight='bold')
        ax1.text(0, 0,
                 'Configuration\nsampleLength : {}s\nmaxPauseLength  : {}s\ntotal '
                 'number of samples : {}'.format(
                     self.sampleLength, self.maxPauseLength,
                     num_positives + num_negatives),
                 style='italic',
                 bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        # Equal aspect ratio ensures that pie is drawn as a circle.
        ax1.axis('equal')
        if save_to:
            plt.savefig(save_to)
        else:
            plt.show()

    @timeit
    def grap_from_video(self, path=None, num_samples=100, **kwargs):
        """
        only to compare the time needed to grap samples from videos to the time needed to load samples from disk.

        Args:
            path (str): Path to samples folder
            numSamples (int): max. number of samples to grab

        Returns:
            samples (list): List of grabbed samples from path
        """
        samples = []
        for sample in self.get_all_samples("faceImage", path, **kwargs):
            samples.append(sample)
            if len(samples) == num_samples:
                break
        return samples

    @timeit
    def grap_from_disk(self, sample_folder, **kwargs):
        """
        only to compare the time needed to grap samples from videos to the time needed to load samples from disk.

        Args:
            sampleFolder (str): Path to samples folder

        Returns:
            samples (list): List of grabbed samples from path
        """
        samples = []
        files = glob.glob(os.path.join(sample_folder, "*.pickle"))
        for file in files:
            s = FeaturedSample()
            s.load(file)
            samples.append(s)
        return samples


def save_balanced_dataset(dataset, save_to, feature_type, data_shape, path=None,
                          ratio_positives=2, ratio_negatives=1,
                          show_status=False, **kwargs):
    """
    saves a balanced dataset to disk

    Args:
        dataset (): complete dataset to process
        saveTo (str): option for additional subfolder path
        featureType (str): feature type to apply on samples
        shape (tuple): shape of samples' images
        path (str): path to store the dataset to
        ratioPositives (int): amount of positive samples to store
        ratioNegatives (int): amount of negative samples to store
        showStatus (bool): Print current status of operation
    """
    positives_folder = os.path.join(save_to, "positiveSamples")
    negatives_folder = os.path.join(save_to, "negativeSamples")
    if not os.path.exists(positives_folder):
        os.makedirs(positives_folder)
    if not os.path.exists(negatives_folder):
        os.makedirs(negatives_folder)

    if show_status:
        # ts = time.perf_counter()
        dataset.debug_print("[getAllPSamples] ###### Status:   0% done")
    if not path:
        path = dataset.path
    folders = list(os.walk(path, followlinks=True))[0][1]
    folders.sort()
    # construct params for producer
    # params = []
    for folder in folders:
        current_folder = os.path.abspath(os.path.join(path, folder))
        # pool.apply_async(producer, producerParams)# #Callback could also be applied
        # producer(dataset, [currentFolder, featureType, shape])
        pool.apply_async(
            producer, (dataset, [current_folder, feature_type, data_shape]))
        # Why does it not start????????
    # start consumer in Thread
    p = multiprocessing.Process(target=consumer, args=(
        positives_folder, negatives_folder, ratio_positives, ratio_negatives))
    print("In multiprocess")

    if __name__ == '__main__':
        multiprocessing.freeze_support()
        p.start()
        # Pool.join()
        pool.close()
        pool.join()
        # kill consumer
        p.terminate()

    self.debug_print(
        "[saveBalancedDataset] Saved balanced dataset! {} samples were droped.".format(
            dataset.dropouts))


def transform_to_hdf5(path, hdf5_path, validation_split=0.2, testing=False):
    """
    transform a pickled dataset to one big hdf5 file.

    Args:
        path (str): path to the folder containing the folders positiveSamples and
            negativeSamples
        hdf5_path (str): folder to where we want to save the hdf5 files
        validation_split (float): Split ratio for validation set
        testing (bool): Validate amount of samples (against fixed amount)
    """
    if not os.path.exists(hdf5_path):
        print('[INFO]: path does not exist - create it')
        os.makedirs(hdf5_path)

    all_pickles = glob.glob(path + '/**/*.pickle', recursive=True)
    if not testing:
        assert len(all_pickles) == 22245 + \
               44489, "You didn't get alle the samples - make sure the path is correct!"

    np.random.shuffle(all_pickles)
    validation_pickles = all_pickles[:int(len(all_pickles) * validation_split)]

    train_pickles = all_pickles[int(len(all_pickles) * validation_split):]
    s = FeatureizedSample()
    s.load(all_pickles[0])
    train_x_shape = (len(train_pickles), *s.get_data().shape)
    train_y_shape = (len(train_pickles),)
    valid_x_shape = (len(validation_pickles), *s.get_data().shape)
    valid_y_shape = (len(validation_pickles),)
    x_dtype = s.get_data().dtype
    y_dtype = np.uint8

    # train
    with h5py.File(os.path.join(hdf5_path, 'vvad_train.hdf5'), mode='w') as hdf5_file:
        hdf5_file.create_dataset('X', shape=train_x_shape, dtype=x_dtype)
        hdf5_file.create_dataset('Y', shape=train_y_shape, dtype=y_dtype)

        for i, sample in enumerate(train_pickles):
            pr = (i / len(train_pickles)) * 100
            print('\r', 'Writing training data: {:.2f}%\r'.format(pr), end='')
            s = FeaturedSample()
            s.load(sample)
            x = s.get_data()
            y = s.get_label()
            hdf5_file['X'][i] = x
            hdf5_file['Y'][i] = y
        print('\r', 'Writing training data: {:.2f}%\r'.format(100.0), end='')
        print()

    # validation
    with h5py.File(os.path.join(hdf5_path, 'vvad_validation.hdf5'), mode='w') as hdf5_file:
        hdf5_file.create_dataset('X', shape=valid_x_shape, dtype=x_dtype)
        hdf5_file.create_dataset('Y', shape=valid_y_shape, dtype=y_dtype)

        for i, sample in enumerate(validation_pickles):
            pr = (i / len(validation_pickles)) * 100
            print(
                '\r', 'Writing validation data: {:.2f}%\r'.format(pr), end='')
            s = FeaturedSample()
            s.load(sample)
            x = s.get_data()
            y = s.get_label()
            hdf5_file['X'][i] = x
            hdf5_file['Y'][i] = y
        print('\r', 'Writing validation data: {:.2f}%\r'.format(100.0), end='')
        print()


def transformPointsToNumpy(points):
    # TODO: this could be faster if we are not using a list at all.
    #  array size is known -> just fill an array
    """
    Transforms given points with x, y coordinates into a numpy array

    Args:
        points (array of dict):  Points to process

    Returns:
        points (numpy array): Given points in one numpy array
    """
    # TODO: this could be faster if we are not using a list at all.
    #  array size is known -> just fill an array
    array = []
    for point in points:
        array.append([point.x, point.y])
    return np.array(array)


# ToDo function not used?? What is it used for?
def transform_to_features(path, shape_model_path=None, shape=None):
    """
    get a Sample of type faceImage and transforms to lipImage, faceFeatures and
    lipFeatures. Saves them in path.

    Args:
        path (str): path to sample
        shapeModelPath (str): path to shape model used by FaceFeatureGenerator
        shape (tuple): shape of images
    """
    ffg = FaceFeatureGenerator(
        "allwfaceImage", shape_model_path=shape_model_path, shape=shape)
    input_sample = FeatureizedSample()
    input_sample.load(path)
    # # get all settings

    lip_images = []
    face_features_list = []
    lip_features_list = []
    # lipImages = np.empty(, dtype=np.uint8)
    for i, frame in enumerate(input_sample.get_data()):
        lip_image, face_features, lip_features = ffg.get_features(
            input_sample.get_data()[i])
        face_features = transform_points_to_numpy(face_features)
        lip_features = transform_points_to_numpy(lip_features)
        lip_images.append(lip_image)
        face_features_list.append(face_features)
        lip_features_list.append(lip_features)

    lip_image_sample = FeatureizedSample()
    lip_image_sample.data = np.array(lip_images)
    lip_image_sample.k = len(lip_images)
    lip_image_sample.label = input_sample.get_label()
    lip_image_sample.featureType = "lipImage"

    face_features_sample = FeatureizedSample()
    face_features_sample.data = np.array(face_features_list)
    face_features_sample.k = len(face_features_list)
    face_features_sample.label = input_sample.get_label()
    face_features_sample.featureType = "faceFeatures"

    lip_features_sample = FeatureizedSample()
    lip_features_sample.data = np.array(lip_features_list)
    lip_features_sample.k = len(lip_features_list)
    lip_features_sample.label = input_sample.get_label()
    lip_features_sample.featureType = "lipFeatures"

    # save the samples in a folder next to the original dataset
    # TODO: extract base path
    path = Path(path)
    # getFilename
    file_name = os.path.join(PurePath(path.parent).name, PurePath(path).name)

    # path for new dataset
    # Assuming the folders positiveSamples and negativeSamples exist
    top_folder = path.parent.parent.parent
    lip_image_folder = os.path.join(top_folder, 'lipImageDataset')
    face_features_folder = os.path.join(top_folder, 'faceFeaturesDataset')
    lip_features_folder = os.path.join(top_folder, 'lipFeaturesDataset')
    pos_neg = ['positiveSamples', 'negativeSamples']
    data_folders = [lip_image_folder, face_features_folder, lip_features_folder]

    for folder in data_folders:
        for subFolder in pos_neg:
            try:
                os.makedirs(os.path.join(folder, subFolder))
            except FileExistsError:
                pass

    lip_image_sample.save(os.path.join(lip_image_folder, file_name))
    face_features_sample.save(os.path.join(face_features_folder, file_name))
    lip_features_sample.save(os.path.join(lip_features_folder, file_name))

    # TODO:call this in a multiproccessing way.


def make_test_set(path, names_path):
    """
    takes the names belonging to the test set from the dataset in path

    Args:
        path (str): Path to the dataset with positiveSamples and negativeSamples
            folder
        namesPath (str): pickleFile with a list of all the fileNames belonging to
            the testset
    """
    test_set_path = os.path.join(path, 'testSet')
    test_pos = os.path.join(test_set_path, 'positiveSamples')
    if not os.path.exists(test_pos):
        os.makedirs(test_pos)
    test_neg = os.path.join(test_set_path, 'negativeSamples')
    if not os.path.exists(test_neg):
        os.makedirs(test_neg)

    with open(names_path, 'rb') as namesFile:
        names = pickle.load(namesFile)

    for name in names:
        print('NAME: {}'.format(name))
        name_path = os.path.join(path, name + ".pickle")
        if os.path.exists(name_path):
            os.rename(name_path, os.path.join(test_set_path, name + '.pickle'))
        else:
            print('NAMEPATH not existing: {}'.format(name_path))
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
        "makeTestSet", "tests", "analyze", "convertAll", "pSamples", "pSamplesAll",
        "samples",
        "download"])  # TODO: remove unused
    parser.add_argument("-d", "--dataPath",
                        help="the path to the dataset", type=str)
    parser.add_argument("-m", "--shapeModelPath",
                        help="the path of folder containing the models", type=str)
    parser.add_argument(
        "--debug", help="debug messages will be displayed", action='store_true')
    parser.add_argument("-p", "--maxPauseLength",
                        help="defines when to break a positive sample in seconds",
                        type=float)
    parser.add_argument("-l", "--sampleLength",
                        help="defines the minimum length of a sample in seconds",
                        type=float)
    parser.add_argument(
        "-f", "--fps", help="the frames per second on the videos", type=int)
    parser.add_argument(
        "-s", "--shape",
        help="[x,y] defines the size to which face or lip images will be resized - "
             "this is the input size of the net",
        type=list)
    parser.add_argument(
        "-n", "--names", help="path to the names pickle file", type=str)

    args = parser.parse_args()

    # get values from config
    config = yaml.load(open(args.config))
    ### Config Values ###
    dataPath = args.dataPath if args.dataPath else config["dataPath"]
    shapeModelPath = args.shapeModelPath if args.shapeModelPath else \
        config["shapeModelPath"]
    debug = args.debug if args.debug else config["debug"]
    maxPauseLength = args.maxPauseLength if args.maxPauseLength else \
        config["maxPauseLength"]
    sampleLength = args.sampleLength if args.sampleLength else config["sampleLength"]
    shape = args.shape if args.shape else config["shape"]
    fps = args.fps if args.fps else config["fps"]

    ds = DataSet(shapeModelPath, debug_flag=debug, sample_length=sampleLength,
                 max_pause_length=maxPauseLength, init_shape=shape, path=dataPath,
                 target_fps=fps)
    if args.option == "download":
        ds.download_lrs3(dataPath)
    if args.option == "pSamples":
        for sample in ds.get_positive_samples(dataPath):
            sample.visualize("mouthImages")
    if args.option == "pSamplesAll":
        for sample in ds.get_all_p_samples(dataPath):
            pass
    if args.option == "convertAll":
        ds.convert_all_fps(dataPath)
    if args.option == "analyze":
        pauses = ds.analyze_negatives(dataPath)
        pSamples = ds.analyze_positives(dataPath)
    if args.option == "tests":
        # samples = ds.grapFromVideo(numSamples=100, dryRun=True)
        # for i, sample in enumerate(samples):
        #     print("Sample number {}".format(i))
        #     assert len(sample) == 2, "SampleConfig needs to have a len of 2 has
        #     {}\nsampleConfig: {}".
        #       format(len(sample), sample)
        #     assert len(sample[1]) == 6, "inner SampleConfig needs to have a len of 6
        #     has {}\ninner sampleConfig: {}".
        #       format(len(sample[1]), sample[1])
        #     print(sample)
        ds.analyze_negatives()


        def test_time():
            logtime_data = {}
            test_folder = "TestPickles"
            if not os.path.exists(test_folder):
                os.makedirs(test_folder)
            samples = ds.grap_from_video(log_time=logtime_data)
            for c, sample in enumerate(samples):
                sample.save(os.path.join(test_folder, str(c) + ".pickle"))
                print("saved sample {}".format(c))

            samples = ds.grap_from_disk(test_folder, log_time=logtime_data)

            print(logtime_data)
        # testTime()
        # saveBalancedDataset(ds, "../data/balancedCleandDataSet/", "faceImage", shape,
        # showStatus=True)
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
        # loadedSample = FeaturedSample()
        # loadedSample.load("testSample.pickle")
        # loadedSample.visualize()

    if args.option == "makeTestSet":
        make_test_set(args.dataPath, args.names)
