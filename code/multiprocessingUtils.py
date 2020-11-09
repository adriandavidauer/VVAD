"""
Utils for multiprocessing
"""
# System imports
import os
import pathlib
import argparse
import shutil
import time
import pickle
import glob
# from collections import deque
import multiprocessing


# 3rd party imports


maxlen = 10
m = multiprocessing.Manager()
positivesQueue = m.Queue(maxsize=maxlen)
negativesQueue = m.Queue(maxsize=maxlen)
sem = multiprocessing.Semaphore(0)
pool = multiprocessing.Pool()



def producer(dataset, getSamplesParams): # positivesQueue, negativesQueue, getSamplesParams, dataset, semaphore
    dataset.debugPrint("started Producer for {}".format(getSamplesParams))
    for sample in dataset.getSamples(*getSamplesParams):
        # Put Samples
        if sample.label:
            if not positivesQueue.full():
                positivesQueue.put(sample) #TODO:raises full Exception https://docs.python.org/2/library/queue.html#Queue.Queue.put
                print("[Producer] puting a positive sample")
            else:
                print("positivesQueue is full. Not puting this positive sample")
        else:
            if not negativesQueue.full():
                negativesQueue.put(sample) #TODO:raises full Exception https://docs.python.org/2/library/queue.html#Queue.Queue.put
                print("[Producer] puting a negative sample")
            else:
                print("negativesQueue is full. Not puting this negative sample")
        if not positivesQueue.empty() and not negativesQueue.empty():
            sem.release()
        # consumer can consume


def consumer(positivesFolder, negativesFolder, ratioPositives, ratioNegatives): # There will be only one consumer, therefore it is thread safe enough
    print("started consumer")
    positiveCounter = 0
    negativeCounter = 0
    savedPositives = 0
    saveNegatives = 0
    while(True):
        print("[CONSUMER] in loop")
        # check ratio and save Samples
        sem.acquire()
        if positiveCounter < ratioPositives and not positivesQueue.empty():
            fname = str(savedPositives) + ".pickle"
            positivesQueue.get().save(os.path.join(positivesFolder, fname))
            print("[CONSUMER] saved sample to {}".format(os.path.join(positivesFolder, fname)))
            savedPositives += 1
            positiveCounter += 1
        if negativeCounter < ratioNegatives and not negativesQueue.empty():
            fname = str(saveNegatives) + ".pickle"
            negativesQueue.get().save(os.path.join(negativesFolder, fname))
            print("[CONSUMER] saved sample to {}".format(os.path.join(negativesFolder, fname)))
            saveNegatives += 1
            negativeCounter += 1
        if negativeCounter == ratioNegatives and positiveCounter == ratioPositives:
            print("[CONSUMER] reseting counters")
            negativeCounter = 0
            positiveCounter = 0


if __name__ == "__main__":
    ##### TEST
    if not positivesQueue.empty():
        print("positivesQueue not empty???")

    if not negativesQueue.empty():
        print("negativesQueue not empty???")
