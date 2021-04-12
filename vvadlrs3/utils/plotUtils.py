import matplotlib.pyplot as plt
import numpy as np
import os
from pymongo import MongoClient
from collections import defaultdict
import pickle
from scipy.interpolate import make_interp_spline, BSpline


def visualizeHistory(history, path=None):
    """
    visualizes the accuracy and loss for training and validation

    :param history: a history from the learninig process
    :type history: dict
    :param path: if set it saves the visualizations to that path
    :type path: string
    """
    loss_values = history['loss']
    val_loss_values = history['val_loss']
    acc = history['acc']
    val_acc = history['val_acc']

    max_val_acc = max(val_acc)
    max_val_acc_arg = np.argmax(val_acc)
    min_val_loss = min(val_loss_values)
    min_val_loss_arg = np.argmin(val_loss_values)

    epochs = range(1, len(loss_values) + 1)
    plt.figure(1)

    plt.plot(epochs, loss_values, 'bo', label='Loss training')
    plt.plot(epochs, val_loss_values, 'b', label='Loss validation')
    plt.hlines(min_val_loss, epochs[0], epochs[-1], colors='r', linestyles='dashed', label='minum loss[{:.2f}] on epoch {}'.format(min_val_loss, min_val_loss_arg + 1))
    plt.title('Values of the loss function for training and validation')
    plt.xlabel('Epochs')
    plt.ylabel('Value of the loss function')
    plt.legend()
    if path:
        plt.savefig(os.path.splitext(path)[0] + '_loss' + os.path.splitext(path)[1])


    plt.figure(2)
    plt.plot(epochs, acc, 'bo', label='Accuracy training')
    plt.plot(epochs, val_acc, 'b', label='Accuracy validation')
    plt.hlines(max_val_acc, epochs[0], epochs[-1], colors='r', linestyles='dashed', label='maximum accuracy[{:.2f} %] on epoch {}'.format(max_val_acc * 100,  max_val_acc_arg + 1))
    plt.title('Accuracy for training and validation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    if path:
        plt.savefig(os.path.splitext(path)[0] + '_accuracy' + os.path.splitext(path)[1])
    
    plt.show()

def calculateHumanAccuracy(mongoURL='mongodb://localhost:27017/', visualize=True, saveTo=None, consider='all'):
    """
    calculates the average accuracy and standard deviation on every sample

    :param mongoURL: the URL to the mongoDB holding the results
    :type mongoURL: String 
    """
    assert consider == 'all' or consider == 'neg' or consider == 'pos', 'consider can only be "all", "pos" or "neg"'
    def calcError(classification, gt):
        return int(classification == gt)
    client = MongoClient(mongoURL)
    data = list(client.humanAccuracy.classifications.find())
    print("I found {} classifications".format(len(data)))
    # Resorting
    samples = defaultdict(list)
    all = []
    for sample in data:
        if consider == 'all':
            color = 'blue'
            samples[sample['sample_num']].append(calcError(sample['classification'], sample['ground_truth']))
            all.append(calcError(sample['classification'], sample['ground_truth']))
        elif consider == 'neg':
            color = 'red'
            if not sample['ground_truth']:
                samples[sample['sample_num']].append(calcError(sample['classification'], sample['ground_truth']))
                all.append(calcError(sample['classification'], sample['ground_truth']))        
        else:
            color = 'green'
            if sample['ground_truth']:
                samples[sample['sample_num']].append(calcError(sample['classification'], sample['ground_truth']))
                all.append(calcError(sample['classification'], sample['ground_truth']))        
    averages = []
    # stds = {}
    y = []
    # e = []
    for key in samples:
        averages.append((key, np.mean(samples[key])))
        y.append(averages[-1][1])
        # stds[key] = np.std(samples[key])
        # e.append(stds[key])

    x = range(len(y))

    # plt.errorbar(x, y, e, linestyle='None', marker='^') #TODO: could look nicer
    if consider == 'neg':
        plt.title('Human Accuracy on test set(only negative samples)')
    elif consider == 'pos':
        plt.title('Human Accuracy on test set(only positive samples)')
    else:
        plt.title('Human Accuracy on test set')
    plt.xlabel('Sample')
    plt.ylabel('Human Accuracy')
    plt.grid(True)
    plt.scatter(x,y, color=color, marker = "_")
    if saveTo:
        plt.savefig(saveTo)
    plt.show()
    # return np.mean(all), np.std(all), averages, stds 
    return np.mean(all), averages

def plotAccOverTimeSteps(histList, path=None, features=False):
    """
    plot the accuracy over different timeDistributed models with different numbers of timeSteps

    :param histList: a list of history files from the learninig process
    :type histList: list of Strings
    :param path: if set it saves the visualizations to that path
    :type path: string
    """
    x = []  # x shows the timesteps/frames used
    y = []  # y shows the reached maximum accuracy
    for histPath in histList:
        # Extract numTimesteps
        shape = histPath.split('(')[1].split(')')[0]
        if len(shape.split(',')) == 3: # that only counts for the images...
            if features:
                numTimesteps = int(shape.split(',')[0])
            else:
                numTimesteps = 1
        elif len(shape.split(',')) == 4:
            numTimesteps = int(shape.split(',')[0])
        else:
            raise ValueError("Couldn't read shape of history file '{}'".format(histFile))
        x.append(numTimesteps)

        # Extract corresponding accuracy
        with open(histPath, 'rb') as history:
            val_accuracies = pickle.load(history)['val_acc']
            y.append(max(val_accuracies))

    x, y = zip(*sorted(zip(x, y)))

    x_new = np.linspace(min(x),max(x),300) #300 represents number of points to make between T.min and T.max

    #print("X.SHAPE: {}".format(x.shape))
    spl = make_interp_spline(x, y,k=3) #BSpline object
    y_smooth = spl(x_new)

    plt.title('Accuracy over the number of used frames')
    plt.xlabel('Frames')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.plot(x_new, y_smooth)
    if path:
        plt.savefig(path)
    plt.show

def plotAccOverImagesize(histList, path=None):
    """
    plot the accuracy over one timeDistributed models with different image sizes

    :param histList: a list of history files from the learninig process
    :type histList: list of Strings
    :param path: if set it saves the visualizations to that path
    :type path: string
    """
    x = []  # x shows the timesteps/frames used
    y = []  # y shows the reached maximum accuracy
    for histPath in histList:
        # Extract image size
        shape = histPath.split('(')[1].split(')')[0]
        if len(shape.split(',')) == 3:
            imagesize = int(shape.split(',')[0])
        elif len(shape.split(',')) == 4:
            imagesize = int(shape.split(',')[1])
        else:
            raise ValueError("Couldn't read shape of history file '{}'".format(histFile))
        x.append(imagesize)

        # Extract corresponding accuracy
        with open(histPath, 'rb') as history:
            val_accuracies = pickle.load(history)['val_acc']
            y.append(max(val_accuracies))

    x, y = zip(*sorted(zip(x, y)))

    x_new = np.linspace(min(x),max(x),300) #300 represents number of points to make between T.min and T.max

    spl = make_interp_spline(x, y,k=3) #BSpline object
    y_smooth = spl(x_new)

    plt.title('Accuracy over the quadractic image size')
    plt.xlabel('Quadractic image size')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.plot(x_new, y_smooth)
    if path:
        plt.savefig(path)
    plt.show