"""
Utils for multiprocessing
"""
# from collections import deque
import multiprocessing
# System imports
import os

# 3rd party imports


maxlen = 10
m = multiprocessing.Manager()
positivesQueue = m.Queue(maxsize=maxlen)
negativesQueue = m.Queue(maxsize=maxlen)
sem = multiprocessing.Semaphore(0)
pool = multiprocessing.Pool()


# positivesQueue, negativesQueue, getSamplesParams, dataset, semaphore
def producer(dataset, get_samples_params):
    """
    The producer extracts positive and negative samples from a given video sample
    and adds the extracted samples to the positive and negative samples queues.

    Args:
        dataset (dataSet): Instance of dataset class
        getSamplesParams (**args): Parameters from a given sample

    """

    dataset.debugPrint("started Producer for {}".format(getSamplesParams))
    for sample in dataset.getSamples(*get_samples_params):
        # Put Samples
        if sample.label:
            if not positivesQueue.full():
                # TODO:raises full Exception https://docs.python.org/2/library/
                #  queue.html#Queue.Queue.put
                positivesQueue.put(sample)
                print("[Producer] putting a positive sample")
            else:
                print("positivesQueue is full. Not puting this positive sample")
        else:
            if not negativesQueue.full():
                # TODO:raises full Exception https://docs.python.org/2/library/
                #  queue.html#Queue.Queue.put
                negativesQueue.put(sample)
                print("[Producer] putting a negative sample")
            else:
                print("negativesQueue is full. Not puting this negative sample")
        if not positivesQueue.empty() and not negativesQueue.empty():
            sem.release()
        # consumer can consume


# There will be only one consumer, therefore it is thread safe enough
def consumer(positives_folder, negatives_folder, ratio_positives, ratio_negatives):
    """
    The consumer consumes all available samples from the positive and negative
    samples queue considering the defined ratio of each sample type.
    Subsequently, these are saved as pickle files on the drive.

    Args:
        positivesFolder (str): path to save positively labeled samples as pickle
            files
        negativesFolder (str): path to save negatively labeled samples as pickle
            files
        ratioPositives (int): amount of positive samples to store
        ratioNegatives (int): amount of negative samples to store

    """
    print("started consumer")
    positive_counter = 0
    negative_counter = 0
    saved_positives = 0
    save_negatives = 0
    while True:
        print("[CONSUMER] in loop")
        # check ratio and save Samples
        sem.acquire()
        if positive_counter < ratio_positives and not positivesQueue.empty():
            fname = str(saved_positives) + ".pickle"
            positivesQueue.get().save(os.path.join(positives_folder, fname))
            print("[CONSUMER] saved sample to {}".format(
                os.path.join(positives_folder, fname)))
            saved_positives += 1
            positive_counter += 1
        if negative_counter < ratio_negatives and not negativesQueue.empty():
            fname = str(save_negatives) + ".pickle"
            negativesQueue.get().save(os.path.join(negatives_folder, fname))
            print("[CONSUMER] saved sample to {}".format(
                os.path.join(negatives_folder, fname)))
            save_negatives += 1
            negative_counter += 1
        if negative_counter == ratio_negatives and positive_counter == ratio_positives:
            print("[CONSUMER] reseting counters")
            negative_counter = 0
            positive_counter = 0


if __name__ == "__main__":
    # TEST
    if not positivesQueue.empty():
        print("positivesQueue not empty???")

    if not negativesQueue.empty():
        print("negativesQueue not empty???")
