"""
Utils for multiprocessing
"""
# ToDo: Are they even used??

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
    # get samples params: path, feature_type, samples_shape, dry_run=False
    dataset.debug_print("started Producer for {}".format(get_samples_params))
    for sample in dataset.get_samples(*get_samples_params):
        # Put Samples
        if sample.label:
            if not positivesQueue.full():
                # TODO:raises full Exception https://docs.python.org/2/library/
                #  queue.html#Queue.Queue.put
                positivesQueue.put(sample)
                print("[Producer] putting a positive sample")
            else:
                print("positivesQueue is full. Not putting this positive sample")
        else:
            if not negativesQueue.full():
                # TODO:raises full Exception https://docs.python.org/2/library/
                #  queue.html#Queue.Queue.put
                negativesQueue.put(sample)
                print("[Producer] putting a negative sample")
            else:
                print("negativesQueue is full. Not putting this negative sample")
        if not positivesQueue.empty() and not negativesQueue.empty():
            sem.release()
        # consumer can consume


# There will be only one consumer, therefore it is thread safe enough
def consumer(positives_folder, negatives_folder, ratio_positives, ratio_negatives):
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
