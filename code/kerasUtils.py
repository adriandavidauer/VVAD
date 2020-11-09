"""
utils needed for keras.
"""
# System imports
import glob
import os
import pickle
import time
#import random
import multiprocessing



# 3rd party imports
import matplotlib.pyplot as plt
import h5py
import numpy as np
import keras
from keras.models import Sequential
from keras.models import load_model
from keras import backend as K
from keras.layers import Dense, Input, Flatten, Dropout, Activation, GlobalAveragePooling2D
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.engine import Model
from keras.optimizers import SGD
from keras.layers.convolutional_recurrent import ConvLSTM2D


from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K

from keras.applications.densenet import DenseNet201, DenseNet121
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet_v2 import MobileNetV2

import cv2






################################ Is not helping - maybe remove later ##########################
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
# Check available GPU devices.
print("The following GPU devices are available: %s" % tf.test.gpu_device_name())
###############################################################################################

from keras_vggface.vggface import VGGFace




# local imports
from sample import *
from timeUtils import *


def splitDataSet(dataPath, ratioTest, randomSeed=42):
    """
    Helper function that returns two lists of pathes to samples - one for training an one for testing.

    :param dataPath: Path to the folder containing two folders - one for negative and one for positive samples
    :type dataPath: String
    :param ratio: the ratio which should be reserved for the test set
    :type ratioTest: float (between 0 and 1)
    :param randomSeed: random seed for reproducable results
    :type randomSeed: int
    """
    trainingData = []
    testData = []

    allPositives = []
    allNegatives = []

    negFolder = os.path.join(dataPath ,  "negativeSamples")
    posFolder = os.path.join(dataPath ,  "positiveSamples")
    assert os.path.exists(negFolder), "No folder {}".format(negFolder)
    assert os.path.exists(posFolder), "No folder {}".format(posFolder)

    allPositives = glob.glob(posFolder + "/*.pickle")
    allNegatives = glob.glob(negFolder + "/*.pickle")

    print('Loaded {} positive samples and {} negative samples'.format(len(allPositives), len(allNegatives)))


    np.random.seed(randomSeed)
    np.random.shuffle(allNegatives)
    np.random.shuffle(allPositives)

    testPositives = allPositives[:int(ratioTest*len(allPositives))]
    trainingPositives = allPositives[int(ratioTest*len(allPositives)):]

    testNegatives = allNegatives[:int(ratioTest*len(allNegatives))]
    trainingNegatives = allNegatives[int(ratioTest*len(allNegatives)):]

    trainingData.extend(trainingPositives)
    trainingData.extend(trainingNegatives)

    testData.extend(testPositives)
    testData.extend(testNegatives)

    return trainingData, testData

class hdf5DataGenerator(keras.utils.Sequence):#keras.utils.Sequence
    'Generates data for Keras'
    def __init__(self, hdf5_path, imageSize = None, num_steps=None, grayscale=False, batch_size=32, randomSeed=42, dataAugmentation=False, shuffle=True, one_hot=False, normalize=False, debug=False):
        '''
        Initialization
        
        :param hdf5_file: hdf_file containing all the data
        :type hdf5_file: hdf_file
        :param imageSize: size of the sample's images
        :type imageSize: tuple of ints
        :param num_steps: number of steps for the sample
        :type num_steps: int
        :param grayscale: decides wheater to use grayscale images or not
        :type grayscale: bool
        :param batch_size: number of samples for the batch
        :type batch_size: int
        :param randomSeed: randomSeed to reproducability
        :type randomSeed: int
        :param dataAugmentation: decides wheater to use data augementation or not
        :type dataAugmentation: bool
        :param shuffle: decides wheater to shuffle the dataset after each epoch
        :type shuffle: bool
        '''
        self.debug = debug
        self.imageSize = imageSize
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.dataAugmentation = dataAugmentation
        self.shuffle = shuffle
        self.grayscale = grayscale
        self.one_hot = one_hot
        self.normalize = normalize
        hdf5_file = h5py.File(hdf5_path, mode='r')
        self.X = hdf5_file['X']
        self.Y = hdf5_file['Y']

        np.random.seed(randomSeed)
        # self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.X.shape[0] / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Get X and Y from index to index+batch
        _X = self.X[index*self.batch_size:(index+1)*self.batch_size]
        _Y = self.Y[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        s = FeatureizedSample()
        s.data = _X[0]
        s.label = _Y[0]
        s.k = _X[0].shape[0]
        x_dtype = _X.dtype
        y_dtype = _Y.dtype
        # Initialization
        X = np.empty((self.batch_size, *s.getData(imageSize=self.imageSize, num_steps=self.num_steps, grayscale=self.grayscale).shape), dtype=x_dtype)
        y = np.empty((self.batch_size), dtype=y_dtype)

        # Generate data
        for i, data in enumerate(zip(_X, _Y)):

            s = FeatureizedSample()
            s.data = data[0]
            s.label = data[1]
            s.k = data[0].shape[0]

            # Store sample
            X[i,] = s.getData(imageSize=self.imageSize, num_steps=self.num_steps, grayscale=self.grayscale)

            # Store class
            y[i] = s.getLabel()
        if self.one_hot:
            return X, keras.utils.to_categorical(y, num_classes=2)
        else:
            return X, y

    # def on_epoch_end(self):
    #     'Updates indexes after each epoch'
    #     if self.shuffle == True:
    #         np.random.shuffle(self.data)


class DataGenerator(keras.utils.Sequence):#keras.utils.Sequence
    'Generates data for Keras'
    def __init__(self, data, imageSize = None, num_steps=None, grayscale=False, batch_size=32, randomSeed=42, dataAugmentation=False, shuffle=True, one_hot=False, normalize=False, debug=False):
        '''
        Initialization
        
        :param data: List of pathes to pickled samples
        :type data: list
        :param imageSize: size of the sample's images
        :type imageSize: tuple of ints
        :param num_steps: number of steps for the sample
        :type num_steps: int
        :param grayscale: decides wheater to use grayscale images or not
        :type grayscale: bool
        :param batch_size: number of samples for the batch
        :type batch_size: int
        :param randomSeed: randomSeed to reproducability
        :type randomSeed: int
        :param dataAugmentation: decides wheater to use data augementation or not
        :type dataAugmentation: bool
        :param shuffle: decides wheater to shuffle the dataset after each epoch
        :type shuffle: bool
        '''
        self.debug = debug
        self.imageSize = imageSize
        self.num_steps = num_steps
        self.data = data
        self.batch_size = batch_size
        self.dataAugmentation = dataAugmentation
        self.shuffle = shuffle
        self.grayscale = grayscale
        self.one_hot = one_hot
        self.normalize = normalize

        np.random.seed(randomSeed)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch = self.data[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X, y = self.__data_generation(batch)
        if self.debug:
            print("Batch with index {} on Process {}".format(index, multiprocessing.current_process()))
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        # if self.shuffle == True:
        #     np.random.shuffle(self.data)
        pass

    def __data_generation(self, batch): 
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # get dims once
        s = FeatureizedSample()
        s.load(batch[0])
        xInit = s.getData(imageSize=self.imageSize, num_steps=self.num_steps, grayscale=self.grayscale, normalize=self.normalize)
        # Initialization
        X = np.empty((self.batch_size, *xInit.shape), dtype=xInit.dtype)

        y = np.empty((self.batch_size), dtype=np.uint8)

        # Generate data
        for i, path in enumerate(batch):

            s = FeatureizedSample()
            s.load(path)
            #s.path = path
            # Store sample
            X[i,] = s.getData(imageSize=self.imageSize, num_steps=self.num_steps, grayscale=self.grayscale, normalize=self.normalize)

            # Store class
            y[i] = s.getLabel()
        if self.one_hot:
            return X, keras.utils.to_categorical(y, num_classes=2)
        else:
            return X, y

class DataGeneratorRAM(keras.utils.Sequence):#keras.utils.Sequence
    'Generates data for Keras'
    def __init__(self, data, imageSize = None, num_steps=None, grayscale=False, batch_size=32, randomSeed=42, dataAugmentation=False, shuffle=True, one_hot=False, normalize=False, debug=False):
        '''
        Initialization
        
        :param data: Tuple of Numpy Arrays in RAM (x,y)
        :type data: Tuple of Numpy Arrays
        :param imageSize: size of the sample's images
        :type imageSize: tuple of ints
        :param num_steps: number of steps for the sample
        :type num_steps: int
        :param grayscale: decides wheater to use grayscale images or not
        :type grayscale: bool
        :param batch_size: number of samples for the batch
        :type batch_size: int
        :param randomSeed: randomSeed to reproducability
        :type randomSeed: int
        :param dataAugmentation: decides wheater to use data augementation or not
        :type dataAugmentation: bool
        :param shuffle: decides wheater to shuffle the dataset after each epoch
        :type shuffle: bool
        '''
        self.debug = debug
        self.imageSize = imageSize
        self.num_steps = num_steps
        self.x = data[0]
        self.y = data[1]
        assert len(self.x) == len(self.y), "X and Y have to be the same size!\nX:{} != Y:{}".format(len(self.x), len(self.y))
        self.batch_size = batch_size
        self.dataAugmentation = dataAugmentation
        self.shuffle = shuffle
        self.grayscale = grayscale
        self.one_hot = one_hot
        self.normalize = normalize

        np.random.seed(randomSeed)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        Xbatch = self.x[index*self.batch_size:(index+1)*self.batch_size]
        y = self.y[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X = self.__data_generation(Xbatch)
        if self.debug:
            print("Batch with index {} on Process {}".format(index, multiprocessing.current_process()))
        if not self.one_hot:
            return X, y
        else:
            return X, keras.utils.to_categorical(y, num_classes=2)

    # def on_epoch_end(self):
    #     'Updates indexes after each epoch'
    #     if self.shuffle == True:
    #         np.random.shuffle(self.data)
    #TODO: is it still shuffling? I expect so

    def __data_generation(self, batch):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        #
        if not self.grayscale and self.num_steps > 1:
            X = np.empty((self.batch_size, self.num_steps, *self.imageSize, 3), dtype=np.uint8)
            for i, origSample in enumerate(batch):
                x = np.empty((self.num_steps, *self.imageSize, 3), dtype=np.uint8)
                for ii, image in enumerate(origSample):
                    x[ii,] = cv2.resize(image, self.imageSize)
                # Store sample
                X[i,] = x
            return X
        else:
            X = np.empty((self.batch_size, *self.imageSize, 3), dtype=np.uint8) #TODO: get this to work with one frame and one_hot
            for i, origSample in enumerate(batch):
                # Store sample
                X[i,] = cv2.resize(origSample, self.imageSize)
            return X

class Models():
    @staticmethod
    def buildFeatureLSTM(input_shape, num_lstm_layers=1, lstm_dims=32, num_dense_layers=1, dense_dims=512, **kwargs):
        model = Sequential()
        #TODO: handle input_shape
        model.add(TimeDistributed(Flatten(input_shape=(input_shape[-2],input_shape[-1]))))
        if num_lstm_layers > 1:
            for i in range(num_lstm_layers - 1):
                # if not i:
                #     model.add(LSTM(lstm_dims, input_shape=input_shape, return_sequences=True))
                #     model.add(BatchNormalization())
                # else:
                model.add(LSTM(lstm_dims, return_sequences=True))
                model.add(BatchNormalization())

        # if model.layers:
        model.add(LSTM(lstm_dims))
        model.add(BatchNormalization())
        # else:
        #     model.add(LSTM(lstm_dims,input_shape=input_shape))
        #     model.add(BatchNormalization())

        ###Add some more dense here
        for i in range(num_dense_layers):
            model.add(Dense(dense_dims, activation='relu'))

        model.add(Dense(1, activation="sigmoid"))
        model.compile(loss="binary_crossentropy", 
                      optimizer='sgd',
                      metrics=["accuracy"])
        
        modelName = 'FeatureLSTM{}_'.format(input_shape) + str(num_lstm_layers) + '_' + str(lstm_dims) + '_' + str(num_dense_layers) + '_' + str(dense_dims)
        return model, modelName

    @staticmethod
    def buildConvLSTM2D(input_shape, kernel_size=(3, 3), filters=40, num_layers = 10, hidden_dense_layers=1, hidden_dense_dim=512, **kwargs):
        chanDim = -1
        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            print("channels_first")
            input_shape = (input_shape[-1], *input_shape[:-1])
            chanDim = 1


        #TODO:Maybe order filters like in VGGFace

        model = Sequential()
        if num_layers > 1:
            for i in range(num_layers - 1):
                model.add(ConvLSTM2D(return_sequences=True, filters=filters, kernel_size=kernel_size, input_shape=input_shape, **kwargs)) #True
                model.add(BatchNormalization())
            model.add(ConvLSTM2D(return_sequences=False, filters=filters, kernel_size=kernel_size, **kwargs)) #True
            model.add(BatchNormalization())
        else:
            model.add(ConvLSTM2D(return_sequences=False, filters=filters, kernel_size=kernel_size, input_shape=input_shape, **kwargs)) #True
            model.add(BatchNormalization())
        
        #TODO: add AveragePooling
        model.add(GlobalAveragePooling2D())

        #model.add(Flatten())
    
        for i in range(hidden_dense_layers):
            model.add(Dense(hidden_dense_dim, activation='relu'))

        model.add(Dense(1, activation="sigmoid"))
        model.compile(loss="binary_crossentropy", 
                      optimizer='rmsprop',
                      metrics=["accuracy"])
        modelName = 'ConvLSTM2D_' + str(num_layers) + '_' + str(filters) + '_' + str(kernel_size)
        return model, modelName

    @staticmethod
    def build0(input_shape, num_classes):#TODO: rename to whichmodel it is - remove num_classes
        model = Sequential()
        chanDim = -1
        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            print("channels_first")
            input_shape = (input_shape[-1], *input_shape[:-1])
            chanDim = 1


        model = Sequential()

        #print("input_shape: {}".format(input_shape))
        #video = Input(shape=input_shape)
        hidden_dim1 = 512
        hidden_dim2 = 128
        #TODO: use the vggFace here(maybe even in resnet structure) https://github.com/rcmalli/keras-vggface#finetuning

        # def freezeLayer(layer):
        #     layer.trainable = False
        #     if hasattr(layer, 'layers'):
        #         for l in layer.layers:
        #             freezeLayer(l)


        vgg_model = VGGFace(include_top=False, input_shape=input_shape[1:])
        # freezeLayer(vgg_model)
        for layer in vgg_model.layers:
            layer.trainable = False
        last_layer = vgg_model.get_layer('pool5').output
        x = Flatten(name='flatten')(last_layer)
        x = Dense(hidden_dim1, activation='relu')(x)
        x = Dense(hidden_dim2, activation='relu')(x)
        custom_vgg_model = Model(vgg_model.input, x)


        ###### TEST ###############
        smallTestCNN = Sequential()
        smallTestCNN.add(Conv2D(32, (3, 3), padding="same",
                        input_shape=input_shape[1:]))
        smallTestCNN.add(Activation("relu"))
        smallTestCNN.add(BatchNormalization(axis=chanDim))
        smallTestCNN.add(MaxPooling2D(pool_size=(2, 2)))
        smallTestCNN.add(Dropout(0.25))
        smallTestCNN.add(Flatten())
        smallTestCNN.add(Dense(512))
        smallTestCNN.add(Activation("relu"))
        smallTestCNN.add(BatchNormalization())
        smallTestCNN.add(Dropout(0.5))
        ###########################



        lstm = LSTM(32)

        output = Dense(num_classes, activation="softmax")

        # model.add(TimeDistributed(smallTestCNN, input_shape=input_shape))
        model.add(TimeDistributed(custom_vgg_model, input_shape=input_shape))
        model.add(lstm)
        model.add(output)

        return model
    @staticmethod
    def buildTimedistributed(base_model_name, num_lstm_layers=1, lstm_dims=32, num_dense_layers=1, dense_dims=512, base_model_weights=None, **kwargs):
        model = Sequential()
        if base_model_name.upper() == "MOBILENET":
            _base_model = MobileNet(weights=None, include_top=False, input_shape=kwargs['input_shape'][1:]) 
            if base_model_weights:
                _base_model.load_weights(base_model_weights)
            base_model_name = "MobileNet"
        elif base_model_name.upper() == "DENSENET":
            _base_model = DenseNet121(weights=None, include_top=False, input_shape=kwargs['input_shape'][1:])
            if base_model_weights:
                _base_model.load_weights(base_model_weights)
            base_model_name = "DenseNet"
        elif base_model_name.upper() == "MOBILENETV2":
            _base_model = MobileNetV2(weights=None, include_top=False, input_shape=kwargs['input_shape'][1:])
            if base_model_weights:
                _base_model.load_weights(base_model_weights)
            base_model_name = "MobileNetV2"
        elif base_model_name.upper() == "VGGFACE":
            _base_model = VGGFace(weights=None, include_top=False, input_shape=kwargs['input_shape'][1:])
            if base_model_weights:
                _base_model.load_weights(base_model_weights)
            base_model_name = "VGGFace"
        flatten = Flatten()(_base_model.output)
        base_model = Model(_base_model.input, flatten)
        model.add(TimeDistributed(base_model, input_shape=kwargs['input_shape']))

        if num_lstm_layers > 1:
            for i in range(num_lstm_layers - 1):
                model.add(LSTM(lstm_dims, return_sequences=True))
                model.add(BatchNormalization())

        model.add(LSTM(lstm_dims))
        model.add(BatchNormalization())

        ###Add some more dense here
        for i in range(num_dense_layers):
            model.add(Dense(dense_dims, activation='relu'))

        model.add(Dense(1, activation="sigmoid"))
        model.compile(loss="binary_crossentropy", 
                      optimizer='sgd',
                      metrics=["accuracy"])
        
        model.summary()
        modelName = 'TimeDistributed{}_'.format(base_model_name) + str(num_lstm_layers) + '_' + str(lstm_dims) + '_' + str(num_dense_layers) + '_' + str(dense_dims)
        return model, modelName

    @staticmethod
    def buildTimedistributedFunctional(base_model_name, num_lstm_layers=1, lstm_dims=32, num_dense_layers=1, dense_dims=512, **kwargs):
        if base_model_name.upper() == "MOBILENET":
            _base_model = MobileNet(weights=None, include_top=False, input_shape=kwargs['input_shape'][1:]) 
        elif base_model_name.upper() == "DENSENET":
            _base_model = DenseNet121(weights=None, include_top=False, input_shape=kwargs['input_shape'][1:])
        flatten = Flatten()(_base_model.output)
        base_model = Model(_base_model.input, flatten)
        input_layer = Input(shape=kwargs['input_shape'])
        x = TimeDistributed(base_model)(input_layer)


        if num_lstm_layers > 1:
            for i in range(num_lstm_layers - 1):
                x = LSTM(lstm_dims, return_sequences=True)(x)
                x = BatchNormalization()(x)

        x = LSTM(lstm_dims)(x)
        x = BatchNormalization()(x)
        #model.add(BatchNormalization())

        ###Add some more dense here
        for i in range(num_dense_layers):
            x = Dense(dense_dims, activation='relu')(x)

        x = Dense(1, activation="sigmoid")(x)

        model = Model(inputs=input_layer, outputs=x)
        model.compile(loss="binary_crossentropy", 
                      optimizer='rmsprop',
                      metrics=["accuracy"])
        modelName = 'TimeDistributedMobileNet_' + str(num_lstm_layers) + '_' + str(lstm_dims) + '_' + str(num_dense_layers) + '_' + str(dense_dims)
        return model, modelName

    @staticmethod
    def buildBaselineModel(base_model_name, **kwargs):

        if base_model_name.upper() == "MOBILENET":
            model = MobileNetV2(**kwargs)
            modelName = 'MobileNet'
        elif base_model_name.upper() == "DENSENET":
            model = DenseNet201(**kwargs)
            modelName = 'DenseNet'
        elif base_model_name.upper() == "VGGFACE":
            model = VGGFace(**kwargs)
            modelName = 'VGGFace'
        elif base_model_name.upper() == "MOBILENETV1":
            model = MobileNet(**kwargs)
            modelName = 'MobileNetV1'
        elif base_model_name.upper() == "DENSENETSMALL":
            model = DenseNet121(**kwargs)
            modelName = 'DenseNetSmall'

        model.compile(loss="categorical_crossentropy", 
                    optimizer='sgd',
                    metrics=["accuracy"])
        return model, modelName



    @staticmethod
    def trainBaselinModel(model, train, test, epochs = 75, batch_size = 32, num_steps = 1, one_hot = False, imageSize=None):
        


        training_generator = DataGenerator(train, num_steps=num_steps, batch_size=batch_size, one_hot=one_hot, imageSize=imageSize)
        validation_generator = DataGenerator(test, num_steps=num_steps, batch_size=batch_size, one_hot=one_hot, imageSize=imageSize)

        history = model.fit_generator(generator=training_generator,
                            validation_data = validation_generator,
                            epochs=epochs,
                            use_multiprocessing=True,
                            workers=8,
                            max_queue_size=16) 
        return history

    @staticmethod
    def saveHistory(history, path):
        with open(path, 'wb') as histFile:
            pickle.dump(history, histFile)
    


def trainModel(model, train, test, epochs, batchSize, initLR=0.01): # TODO:every model will be trained sligthly different because of different samples
    """
    trains the given model with the given params and data
    """

    # initialize the model and optimizer (you'll want to use
    # binary_crossentropy for 2-class classification)
    print("[INFO] training network...")
    opt = SGD(lr=initLR, decay=initLR / epochs)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
    	metrics=["accuracy"])                                       #TODO: should be compiled already!

    training_generator = DataGenerator(train)
    validation_generator = DataGenerator(test)


    # for i in range(len(training_generator)):
    #     batch = training_generator[i]
    #     model.train_on_batch(*batch)


    model.fit_generator(generator=training_generator,
                        validation_data = validation_generator,
                        epochs=epochs,
                        use_multiprocessing=False,
                        workers=5,
                        max_queue_size=5) # should end up in appr. 5MB * 32(batch_size) * 5(workers) * 5(max_queue_size) = 4GB on RAM

def genData(train, test, train_samples=None, valid_samples= None, print_freq=None,  **kwargs):
    """
    Put the whole data or just some batches.
    """

    #Put batch_size to 1
    kwargs_batch_size_1 = dict(kwargs)
    kwargs_batch_size_1['batch_size'] = 1
    t = DataGenerator(train , **kwargs_batch_size_1)
    v = DataGenerator(test , **kwargs_batch_size_1)

    if train_samples:
        trainSampleList = range(train_samples)
    else:
        trainSampleList = range(len(t))


    if valid_samples:
        validSampleList = range(valid_samples)
    else:
        validSampleList = range(len(v))



    dtype = t[0][0].dtype

    train_x_shape = (len(trainSampleList), *t[0][0].shape[1:])
    train_y_shape = (len(trainSampleList), *t[0][1].shape[1:])

    valid_x_shape = (len(validSampleList), *v[0][0].shape[1:])
    valid_y_shape = (len(validSampleList), *v[0][1].shape[1:])

    #calculate needed memory:
    memory_bytes = (dtype.itemsize)  * (np.prod(train_x_shape) + np.prod(train_y_shape) + np.prod(valid_x_shape) + np.prod(valid_y_shape))


    print("train_x_shape: {}".format(train_x_shape))
    print("train_y_shape: {}".format(train_y_shape))
    print("valid_x_shape: {}".format(valid_x_shape))
    print("valid_y_shape: {}".format(valid_y_shape))
    print("dtype: {}".format(dtype))
    print("{} GB RAM is needed for training and validation data".format(memory_bytes / (1024 * 1024 * 1024) ))

    train_x = np.empty(train_x_shape, dtype=dtype)
    train_y = np.empty(train_y_shape, dtype=dtype)
    vali_x = np.empty(valid_x_shape, dtype=dtype)
    vali_y = np.empty(valid_y_shape, dtype=dtype)

    start = None
    end = None
    new = 0
    avg = 0
    eta = None
    for x in trainSampleList:
        start = time.time()
        pr = (x / len(trainSampleList))*100
        if not print_freq:
            print ('\r', 'Training data: {:.2f}%\tETA: {}s\r'.format(pr, eta),end='')
        else:
            if not x % print_freq:
                print('Training data: {:.2f}%\tETA: {}s\r'.format(pr, eta))
        train_x[x] = t[x][0][0]
        train_y[x] = t[x][1][0]
        end = time.time()
        new = end - start
        avg = (avg * x + new ) / (x + 1)
        eta = (avg) * (len(trainSampleList) - (1 + x))
    print ('\r', 'Training data: {:.2f}%\r'.format(100.0),end='')
    print()

    start = None
    end = None
    eta = None
    for x in validSampleList:
        start = time.time()
        pr = (x / len(validSampleList))*100
        if not print_freq:
            print ('\r', 'Validation data: {:.2f}%\tETA: {}s\r'.format(pr, eta),end='')
        else:
            if not x % print_freq:
                print('Validation data: {:.2f}%\tETA: {}s\r'.format(pr, eta))
        vali_x[x] = v[x][0][0]
        vali_y[x] = v[x][1][0]
        end = time.time()
        new = end - start
        avg = (avg * x + new ) / (x + 1)
        eta = (avg) * (len(trainSampleList) - (1 + x))
    print ('\r', 'Validation data: {:.2f}%\r'.format(100.0),end='')
    print()

    return (train_x, train_y), (vali_x, vali_y)


def genDataInternal(train, test, train_samples=None, valid_samples= None, print_freq=None,  **kwargs):
    """
    Put the whole data or just some batches. - train, test are tuoples of numpyArrays in this case
    """

    #Put batch_size to 1
    kwargs_batch_size_1 = dict(kwargs)
    kwargs_batch_size_1['batch_size'] = 1
    t = DataGeneratorRAM(train , **kwargs_batch_size_1)
    v = DataGeneratorRAM(test , **kwargs_batch_size_1)

    if train_samples:
        trainSampleList = range(train_samples)
    else:
        trainSampleList = range(len(t))


    if valid_samples:
        validSampleList = range(valid_samples)
    else:
        validSampleList = range(len(v))



    dtype = t[0][0].dtype

    train_x_shape = (len(trainSampleList), *t[0][0].shape[1:])
    train_y_shape = (len(trainSampleList), *t[0][1].shape[1:])

    valid_x_shape = (len(validSampleList), *v[0][0].shape[1:])
    valid_y_shape = (len(validSampleList), *v[0][1].shape[1:])

    train_x = np.empty(train_x_shape, dtype=dtype)
    train_y = np.empty(train_y_shape, dtype=dtype)
    vali_x = np.empty(valid_x_shape, dtype=dtype)
    vali_y = np.empty(valid_y_shape, dtype=dtype)

    start = None
    end = None
    new = 0
    avg = 0
    eta = None
    for x in trainSampleList:
        start = time.time()
        pr = (x / len(trainSampleList))*100
        if not print_freq:
            print ('\r', 'Training data: {:.2f}%\tETA: {}s\r'.format(pr, eta),end='')
        else:
            if not x % print_freq:
                print('Training data: {:.2f}%\tETA: {}s\r'.format(pr, eta))
        train_x[x] = t[x][0][0]
        train_y[x] = t[x][1][0]
        end = time.time()
        new = end - start
        avg = (avg * x + new ) / (x + 1)
        eta = (avg) * (len(trainSampleList) - (1 + x))
    print ('\r', 'Training data: {:.2f}%\r'.format(100.0),end='')
    print()

    start = None
    end = None
    eta = None
    for x in validSampleList:
        start = time.time()
        pr = (x / len(validSampleList))*100
        if not print_freq:
            print ('\r', 'Validation data: {:.2f}%\tETA: {}s\r'.format(pr, eta),end='')
        else:
            if not x % print_freq:
                print('Validation data: {:.2f}%\tETA: {}s\r'.format(pr, eta))
        vali_x[x] = v[x][0][0]
        vali_y[x] = v[x][1][0]
        end = time.time()
        new = end - start
        avg = (avg * x + new ) / (x + 1)
        eta = (avg) * (len(trainSampleList) - (1 + x))
    print ('\r', 'Validation data: {:.2f}%\r'.format(100.0),end='')
    print()

    return (train_x, train_y), (vali_x, vali_y)


@timeit
def hdf5SamplesToMemory(train_path, val_path, train_samples=None, valid_samples= None,  **kwargs):
    
    kwargs_batch_size_1 = dict(kwargs)
    kwargs_batch_size_1['batch_size'] = 1
    t = hdf5DataGenerator(train_path, **kwargs_batch_size_1)
    v = hdf5DataGenerator(val_path, **kwargs_batch_size_1)
    
    if train_samples:
        trainSampleList = range(train_samples)
    else:
        trainSampleList = range(len(t))


    if valid_samples:
        validSampleList = range(valid_samples)
    else:
        validSampleList = range(len(v))

    dtype = t[0][0].dtype

    train_x_shape = (len(trainSampleList), *t[0][0].shape[1:])
    train_y_shape = (len(trainSampleList), *t[0][1].shape[1:])

    valid_x_shape = (len(validSampleList), *v[0][0].shape[1:])
    valid_y_shape = (len(validSampleList), *v[0][1].shape[1:])

    train_x = np.empty(train_x_shape, dtype=dtype)
    train_y = np.empty(train_y_shape, dtype=dtype)
    vali_x = np.empty(valid_x_shape, dtype=dtype)
    vali_y = np.empty(valid_y_shape, dtype=dtype)


    for x in trainSampleList:
        pr = (x / len(trainSampleList))*100
        # print ('\r', 'Training data: {:.2f}%\r'.format(pr),end='')
        train_x[x] = t[x][0][0]
        train_y[x] = t[x][1][0]
    # print ('\r', 'Training data: {:.2f}%\r'.format(100.0),end='')
    # print()

    for x in validSampleList:
        pr = (x / len(validSampleList))*100
        # print ('\r', 'Validation data: {:.2f}%\r'.format(pr),end='')
        vali_x[x] = v[x][0][0]
        vali_y[x] = v[x][1][0]
    # print ('\r', 'Validation data: {:.2f}%\r'.format(100.0),end='')
    # print()

    return (train_x, train_y), (vali_x, vali_y)


def get_model_memory_usage(batch_size, model): #https://stackoverflow.com/questions/43137288/how-to-determine-needed-memory-of-keras-model
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    number_size = 4.0
    if K.floatx() == 'float16':
         number_size = 2.0
    if K.floatx() == 'float64':
         number_size = 8.0

    total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes

@timeit
def checkDataGen(dataGen, var):
    var = dataGen[0]

def testModel(model_path, test_set_path, saveTo = None):
    """
    Test a specific model with the corresponding test set.
    Maybe even plots every sample - numbers will probably dont correspond to numbers in humAccTest

    :param model_path: Path to the model
    :type model_path: String
    :param test_set_path: path to the corresponding test set
    :type test_set_path: String
    :return: the accuracy, (mae, std), (mse, std),  list of samples that were wrong classifies by the model
    :rtype: float, tuple of floats, tuple of floats, list of Stings
    """
    correctClassifications = []
    errors = []
    percentages = {}
    model = load_model(model_path)

    # negFolder = os.path.join(test_set_path, "negativeSamples")
    # posFolder = os.path.join(test_set_path, 'positiveSamples')
    normalize = False
    imageSize = None
    num_steps = model.input_shape[1]

    if 'Features' in test_set_path:
        # FEATURES NEED TO BE NORMALIZED!!!
        normalize = True
    else:
        # Imagesize needs to be set
        if 'lip' in test_set_path:
            # Bug with non quadratic imagesizes :/
            imageSize = None
        else:
            imageSize = model.input_shape[-3:-1]
        # imageSize = (model.input_shape[-2], model.input_shape[-3])

    imageSize = None ###HACK
    print ('num_steps: {}'.format(num_steps))
    print("imageSize: {}".format(imageSize))
    print("normalize: {}".format(normalize))
    yList = []
    eList = []
    lolims = []
    uplims = []
    y_percents = []
    xList = []
    for i, samplePath in enumerate(glob.glob(os.path.join(os.path.join(test_set_path, '**'), '*.pickle'))):
        sample = FeatureizedSample()
        sample.load(samplePath)
        data = sample.getData(normalize=normalize, imageSize=imageSize, num_steps=num_steps)
        # v = FeatureizedSample()
        # v.data = data
        # v.featureType = "faceImage"
        # v.visualize(saveTo=str(i) + '.gif')   #This seems to be okay.
        # make it a list of samples with only that one sample...
        x = np.empty((1, *data.shape))
        x[0] = data
        label = sample.getLabel()
        y_percent = model.predict(x)[0][0]
        y_percents.append(y_percent)
        yList.append(label)
        eList.append(abs(label - y_percent))
        uplims.append(label)
        lolims.append(not label)
        percentages[samplePath] = (label, y_percent)
        y = np.rint(y_percent)
        xList.append(samplePath.split('/')[-1].split('.')[0])
        if y != label:
            errors.append(samplePath)


    #print("Predictions: {}".format(y_percents))
    #plot the samples with the error between label and prediction
    #xList = list(range(len(yList)))#HACK
    plt.title('Classifications on the test set')
    plt.xlabel('Sample')
    plt.ylabel('Classification')
    plt.grid(True)
    plt.errorbar(xList, yList, eList, linestyle='None', lolims=lolims, uplims=uplims) 
    plt.hlines(0.5, xList[0], xList[-1], colors='r', linestyles='dashed', label='Decision Boundary')
    if saveTo:
        plt.savefig(saveTo)
    plt.show()

    mae = np.mean(eList)
    maeStd = np.std(eList)
    mse = np.mean(np.square(eList))
    mseStd = np.std(np.square(eList))
    return (((i+1) - len(errors)) / (i+1),(mae, maeStd), (mse, mseStd),  errors)


if __name__ == "__main__":
    print('############################START###########################')
    models = glob.glob('*model-*.h5')
    print ('[MODELS]: {}'.format(models))
    testset = '/gluster/scratch/alubitz/balancedCleandDataSet/testSet/'
    for model in models:
        acc, (mae, maeStd), (mse, mseStd),  errors = testModel(model, testset)
        print("Accuracy for {} is {}".format(model.split("/")[-1], acc))
        print("MAE for {} is {} with std {}".format(model.split("/")[-1], mae, maeStd))
