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
def producer(dataset, getSamplesParams):
    """
    The producer extracts positive and negative samples from a given video sample
    and adds the extracted samples to the positive and negative samples queues.

    Args:
        dataset (dataSet): Instance of dataset class
        getSamplesParams (**args): Parameters from a given sample

    """

    dataset.debugPrint("started Producer for {}".format(getSamplesParams))
    for sample in dataset.getSamples(*getSamplesParams):
        # Put Samples
        if sample.label:
            if not positivesQueue.full():
                # TODO:raises full Exception https://docs.python.org/2/library/queue.html#Queue.Queue.put
                positivesQueue.put(sample)
                print("[Producer] putting a positive sample")
            else:
                print("positivesQueue is full. Not puting this positive sample")
        else:
            if not negativesQueue.full():
                # TODO:raises full Exception https://docs.python.org/2/library/queue.html#Queue.Queue.put
                negativesQueue.put(sample)
                print("[Producer] putting a negative sample")
            else:
                print("negativesQueue is full. Not puting this negative sample")
        if not positivesQueue.empty() and not negativesQueue.empty():
            sem.release()
        # consumer can consume


# There will be only one consumer, therefore it is thread safe enough
def consumer(positivesFolder, negativesFolder, ratioPositives, ratioNegatives):
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

    positiveCounter = 0
    negativeCounter = 0
    savedPositives = 0
    saveNegatives = 0
    while True:
        print("[CONSUMER] in loop")
        # check ratio and save Samples
        sem.acquire()
        if positiveCounter < ratioPositives and not positivesQueue.empty():
            fname = str(savedPositives) + ".pickle"
            positivesQueue.get().save(os.path.join(positivesFolder, fname))
            print("[CONSUMER] saved sample to {}".format(
                os.path.join(positivesFolder, fname)))
            savedPositives += 1
            positiveCounter += 1
        if negativeCounter < ratioNegatives and not negativesQueue.empty():
            fname = str(saveNegatives) + ".pickle"
            negativesQueue.get().save(os.path.join(negativesFolder, fname))
            print("[CONSUMER] saved sample to {}".format(
                os.path.join(negativesFolder, fname)))
            saveNegatives += 1
            negativeCounter += 1
        if negativeCounter == ratioNegatives and positiveCounter == ratioPositives:
            print("[CONSUMER] reseting counters")
            negativeCounter = 0
            positiveCounter = 0


if __name__ == "__main__":
    # TEST
    if not positivesQueue.empty():
        print("positivesQueue not empty???")

    if not negativesQueue.empty():
        print("negativesQueue not empty???")
