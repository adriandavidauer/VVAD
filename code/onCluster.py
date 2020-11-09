'''To run on the DFKI-cluster'''
# System imports
import argparse
import os
# 3rd Party imports
from keras import callbacks
from keras.models import load_model

# local imports
from kerasUtils import *
from plotUtils import *
import datetime
import time
# end file header
__author__      = 'Adrian Lubitz'
__copyright__   = 'Copyright (c)2017, Blackout Technologies'

#dataPath = '/gluster/scratch/alubitz/balancedCleandDataSet'
dataPath = '/gluster/scratch/alubitz/lipImageDataset'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", help='what should be done' , type=str, choices=['timeSteps', 'imageSize', 'bestLipEndToEnd', 'bestFaceEndToEnd', 'FeatureTimeSteps', 'bestFeatures', 'bestFaceEndToEndFromCheckpoint'])
    parser.add_argument("-m", "--base_model",help="name of the base model", type=str)
    parser.add_argument("-s", "--step_size",help="step size for the timeSteps or imageSize", type=int)
    parser.add_argument("-f", "--frames", help="Number of frames to evaluate the image size on", type=int)
    parser.add_argument("-g", "--start",help="start from this number of frames", type=int)
    parser.add_argument("-d", "--dataPath",help="path to the dataset", type=str)
    parser.add_argument("-n", "--modelName",help="path to the location where the best model should be saved", type=str)




    args = parser.parse_args()

    if args.dataPath:
        dataPath = args.dataPath

    if args.task == 'timeSteps':
        print("start: {}".format(datetime.datetime.now()))
        # Get all sample pathes
        train, test = splitDataSet(dataPath, 0.15, 42) # 1 to 1 Dataset
        print('Loaded {} training samples and {} validation samples'.format(len(train), len(test)))

        # Load samples to RAM
        batch_size = 64
        #imageSize = (96, 96) # faces
        imageSize = None # lips
        grayscale = False
        num_steps = 38 # 38 is max 
        one_hot = False
        train_samples= None 
        valid_samples= None  


        (train_x, train_y), validation_data = genData(train, 
                                                    test,train_samples=train_samples, 
                                                    valid_samples= valid_samples, 
                                                    batch_size = batch_size, 
                                                    imageSize = imageSize,
                                                    grayscale = grayscale, 
                                                    num_steps=num_steps, 
                                                    one_hot=one_hot,
                                                    print_freq=1000
                                                    )

        # Generate list of num_timesteps to use
        start = args.start
        step_size = args.step_size
        maxi = 38
        timeSteps = list(range(start,maxi,step_size))
        #ensure to have the max in the list
        if not timeSteps[-1] == maxi:
            timeSteps.append(maxi)
        if timeSteps[0] == 1:
            timeSteps.pop(0) 
        print("Evaluatiing for {} frames".format(timeSteps))
        for timeStep in timeSteps:
            train_x_parted = train_x[:, :timeStep]
            train_y_parted = train_y
            validation_data_parted = (validation_data[0][:,:timeStep], validation_data[1])

            input_shape = train_x_parted[0].shape
            print(input_shape)
            epochs = 200
            model, modelName = Models.buildTimedistributed(args.base_model, 
                                                        input_shape=input_shape, 
                                                        #lstm_dims = 128, 
                                                        #num_lstm_layers= 4, 
                                                        #num_dense_layers=0,
                                                        #dense_dims=512,
                                                        #base_model_weights='../models/mobileNetBaselineWeights.h5'
                                                        )

            history = model.fit(x=train_x_parted, y=train_y_parted, validation_data=validation_data_parted, batch_size=batch_size, epochs=epochs)
            num_samples = 'all' if not train_samples else train_samples
            Models.saveHistory(history.history, "../trainingHistories/onCluster_{}_{}_{}_{}_{}.pickle".format(num_samples, modelName, batch_size, input_shape, datetime.datetime.now()))

    elif args.task == 'imageSize': 
        print("start: {}".format(datetime.datetime.now()))
        # Get all sample pathes
        train, test = splitDataSet(dataPath, 0.15, 42) # 1 to 1 Dataset
        print('Loaded {} training samples and {} validation samples'.format(len(train), len(test)))
        
        # Load samples to RAM
        batch_size = 64
        imageSize = (200, 200) #MAX
        grayscale = False
        num_steps = 1
        one_hot = False
        train_samples= None 
        valid_samples= None 
        print_freq=1000
        epochs = 200
        start = time.time()
        (train_x, train_y), validation_data = genData(train, 
                                                        test,train_samples=train_samples, 
                                                        valid_samples= valid_samples, 
                                                        batch_size = batch_size, 
                                                        imageSize = imageSize,
                                                        grayscale = grayscale, 
                                                        num_steps=num_steps, 
                                                        one_hot=one_hot,
                                                        print_freq=print_freq
                                                        )
        print("Needed {}s to load everything in RAM".format(time.time() - start))
        input_shape = train_x[0].shape
        output_shape = train_y.shape
        print("INIT INPUTSHAPE: {}".format(input_shape))
        print("INIT OUTPUTSHAPE: {}".format(output_shape))

        # Generate list of image_sizes to use
        start = args.start
        step_size = args.step_size
        maxi = 200
        image_sizes = list(range(start,maxi,step_size))
        #ensure to have the max in the list
        if not image_sizes[-1] == maxi:
            image_sizes.append(maxi)

        for image_size in image_sizes:
            start = time.time()
            (train_x_resized, train_y_resized), validation_data_resized = genDataInternal((train_x, train_y), 
                                            validation_data,train_samples=train_samples, 
                                            valid_samples= None, 
                                            batch_size = None, 
                                            imageSize = (image_size, image_size),
                                            grayscale = grayscale, 
                                            num_steps=num_steps, 
                                            one_hot=True,
                                            print_freq=1000
                                            )

            input_shape = train_x_resized[0].shape
            output_shape = train_y_resized[0].shape
            print("INPUTSHAPE: {}".format(input_shape))
            print("OUTPUTSHAPE: {}".format(output_shape))
            model, modelName = Models.buildBaselineModel('mobileNet', 
                                                        input_shape=input_shape, weights=None, classes=2,
                                                        #lstm_dims = 128, 
                                                        #num_lstm_layers= 4, 
                                                        #num_dense_layers=0,
                                                        #dense_dims=512
                                                        )

            history = model.fit(x=train_x_resized, y=train_y_resized, validation_data=validation_data_resized, batch_size=batch_size, epochs=epochs)
            num_samples = 'all' if not train_samples else train_samples
            Models.saveHistory(history.history, "../trainingHistories/IMAGEEVAL{}_{}_{}_{}_{}.pickle".format(num_samples, modelName, batch_size, input_shape, datetime.datetime.now()))
            print("Needed {}s to train a model".format(time.time() - start))



            # training_generator = DataGeneratorRAM((train_x, train_y), imageSize = (image_size, image_size), num_steps=num_steps, batch_size=64)
            # validation_generator = DataGeneratorRAM(validation_data, imageSize = (image_size, image_size), num_steps=num_steps, batch_size=64)
            # input_shape = training_generator[0][0][0].shape
            # print(input_shape)
            # model, modelName = Models.buildTimedistributed('mobileNet', 
            #                                             input_shape=input_shape, 
            #                                             #lstm_dims = 128, 
            #                                             #num_lstm_layers= 4, 
            #                                             #num_dense_layers=0,
            #                                             #dense_dims=512
            #                                             )

            # history = model.fit_generator(generator=training_generator,
            #                     validation_data=validation_generator,
            #                     use_multiprocessing=True,
            #                     workers=max([multiprocessing.cpu_count() - 5, 10]), epochs=epochs)
            # num_samples = 'all' if not train_samples else train_samples
            # Models.saveHistory(history.history, "../trainingHistories/onClustercheckImageSize{}_{}_{}_{}_{}.pickle".format(num_samples, modelName, batch_size, input_shape, datetime.datetime.now()))

    elif args.task == 'bestFaceEndToEnd':
        print("start: {}".format(datetime.datetime.now()))
        # Get all sample pathes
        train, test = splitDataSet(dataPath, 0.15, 42) # 1 to 1 Dataset
        print('Loaded {} training samples and {} validation samples'.format(len(train), len(test)))

        # Load samples to RAM
        batch_size = 64
        imageSize = (96, 96) # faces
        # imageSize = None # lips
        grayscale = False
        num_steps = 36 # 38 is max 
        one_hot = False
        train_samples= None 
        valid_samples= None  


        (train_x, train_y), validation_data = genData(train, 
                                                    test,train_samples=train_samples, 
                                                    valid_samples= valid_samples, 
                                                    batch_size = batch_size, 
                                                    imageSize = imageSize,
                                                    grayscale = grayscale, 
                                                    num_steps=num_steps, 
                                                    one_hot=one_hot,
                                                    print_freq=1000
                                                    )
        input_shape = train_x[0].shape
        print(input_shape)
        epochs = 200
        baseModel = "MobileNet"
        model, modelName = Models.buildTimedistributed(baseModel, 
                                                    input_shape=input_shape, 
                                                    #lstm_dims = 128, 
                                                    #num_lstm_layers= 4, 
                                                    #num_dense_layers=0,
                                                    #dense_dims=512,
                                                    #base_model_weights='../models/mobileNetBaselineWeights.h5'
                                                    )

        # define callbacks
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.1,
                              patience=10, min_lr=0.001, cooldown=2)
        earlyStopping = callbacks.EarlyStopping(monitor='val_acc', patience=30, restore_best_weights=True)

        checkpoint = callbacks.ModelCheckpoint(args.task + ".h5", monitor='val_acc', save_best_only=True)

        history = model.fit(x=train_x, y=train_y, validation_data=validation_data, batch_size=batch_size, epochs=epochs, callbacks=[reduce_lr, earlyStopping, checkpoint])
        num_samples = 'all' if not train_samples else train_samples
        Models.saveHistory(history.history, "../trainingHistories/onCluster_{}_{}_{}_{}_{}_{}.pickle".format(args.task, num_samples, modelName, batch_size, input_shape, datetime.datetime.now()))
        

    elif args.task == 'FeatureTimeSteps':
        print("start: {}".format(datetime.datetime.now()))
        # Get all sample pathes
        train, test = splitDataSet(dataPath, 0.15, 42) # 1 to 1 Dataset
        print('Loaded {} training samples and {} validation samples'.format(len(train), len(test)))

        # Load samples to RAM
        batch_size = 64
        #imageSize = (96, 96) # faces
        imageSize = None # lips
        grayscale = False
        num_steps = 38 # 38 is max 
        one_hot = False
        train_samples= None 
        valid_samples= None  
        normalize = True
        print_freq=1000


        (train_x, train_y), validation_data = genData(train, 
                                                    test,train_samples=train_samples, 
                                                    valid_samples= valid_samples, 
                                                    batch_size = batch_size, 
                                                    imageSize = imageSize,
                                                    grayscale = grayscale, 
                                                    num_steps=num_steps, 
                                                    one_hot=one_hot,
                                                    print_freq=print_freq, 
                                                    normalize=normalize
                                                    )

        # Generate list of num_timesteps to use
        start = args.start
        step_size = args.step_size
        maxi = 38
        timeSteps = list(range(start,maxi,step_size))
        #ensure to have the max in the list
        if not timeSteps[-1] == maxi:
            timeSteps.append(maxi)
        if timeSteps[0] == 1:
            timeSteps.pop(0) 
        print("Evaluatiing for {} frames".format(timeSteps))
        for timeStep in timeSteps:
            train_x_parted = train_x[:, :timeStep]
            train_y_parted = train_y
            validation_data_parted = (validation_data[0][:,:timeStep], validation_data[1])

            input_shape = train_x_parted[0].shape
            print(input_shape)
            epochs = 200
            model, modelName = Models.buildFeatureLSTM(input_shape=input_shape)

            history = model.fit(x=train_x_parted, y=train_y_parted, validation_data=validation_data_parted, batch_size=batch_size, epochs=epochs)
            num_samples = 'all' if not train_samples else train_samples
            Models.saveHistory(history.history, "../trainingHistories/onClusterFeatureLSTM_{}_{}_{}_{}_{}.pickle".format(num_samples, modelName, batch_size, input_shape, datetime.datetime.now()))

    elif args.task == 'bestFeatures':
        print("start: {}".format(datetime.datetime.now()))
        # Get all sample pathes
        train, test = splitDataSet(dataPath, 0.15, 42) # 1 to 1 Dataset
        print('Loaded {} training samples and {} validation samples'.format(len(train), len(test)))

        # Load samples to RAM
        batch_size = 64
        #imageSize = (96, 96) # faces
        imageSize = None # lips or features
        grayscale = False
        num_steps = 38 # 38 is max 
        one_hot = False
        train_samples= None 
        valid_samples= None  
        normalize = True
        print_freq=1000


        (train_x, train_y), validation_data = genData(train, 
                                                    test,train_samples=train_samples, 
                                                    valid_samples= valid_samples, 
                                                    batch_size = batch_size, 
                                                    imageSize = imageSize,
                                                    grayscale = grayscale, 
                                                    num_steps=num_steps, 
                                                    one_hot=one_hot,
                                                    print_freq=print_freq, 
                                                    normalize=normalize
                                                    )

        input_shape = train_x[0].shape
        print(input_shape)
        epochs = 200
        model, modelName = Models.buildFeatureLSTM(input_shape=input_shape)

        # define callbacks
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.1,
                              patience=10, min_lr=0.001, cooldown=2)
        earlyStopping = callbacks.EarlyStopping(monitor='val_acc', patience=30, restore_best_weights=True)

        checkpoint = callbacks.ModelCheckpoint(args.modelName + ".h5", monitor='val_acc', save_best_only=True)

        callbacks=[checkpoint]

        history = model.fit(x=train_x, y=train_y, validation_data=validation_data, batch_size=batch_size, epochs=epochs, callbacks=callbacks)
        num_samples = 'all' if not train_samples else train_samples
        Models.saveHistory(history.history, "../trainingHistories/onCluster_{}_{}_{}_{}_{}_{}.pickle".format(args.task, num_samples, modelName, batch_size, input_shape, datetime.datetime.now()))

        modelPath = args.modelName + ".h5"
        testSet = os.path.join(dataPath, "testSet")
        acc,(mae, maeStd), (mse, mseStd), errors = testModel(modelPath, testSet)
        print("Accuracy for {} is {}".format(modelPath.split("/")[-1], acc))
        print("MAE for {} is {} with std {}".format(modelPath.split("/")[-1], mae, maeStd))                                           

    elif args.task == 'bestLipEndToEnd':
        print("start: {}".format(datetime.datetime.now()))
        # Get all sample pathes
        train, test = splitDataSet(dataPath, 0.15, 42) # 1 to 1 Dataset
        print('Loaded {} training samples and {} validation samples'.format(len(train), len(test)))

        # Load samples to RAM
        batch_size = 64
        imageSize = None#(160, 160) # faces
        # imageSize = None # lips
        grayscale = False
        num_steps = 38 # 38 is max 
        one_hot = False
        train_samples= None 
        valid_samples= None  


        (train_x, train_y), validation_data = genData(train, 
                                                    test,train_samples=train_samples, 
                                                    valid_samples= valid_samples, 
                                                    batch_size = batch_size, 
                                                    imageSize = imageSize,
                                                    grayscale = grayscale, 
                                                    num_steps=num_steps, 
                                                    one_hot=one_hot,
                                                    print_freq=1000
                                                    )
        input_shape = train_x[0].shape
        print(input_shape)
        epochs = 200
        baseModel = "MobileNet"
        model, modelName = Models.buildTimedistributed(baseModel, 
                                                    input_shape=input_shape, 
                                                    #lstm_dims = 128, 
                                                    #num_lstm_layers= 4, 
                                                    #num_dense_layers=0,
                                                    #dense_dims=512,
                                                    #base_model_weights='../models/mobileNetBaselineWeights.h5'
                                                    )

        # define callbacks
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.1,
                              patience=10, min_lr=0.001, cooldown=2)
        earlyStopping = callbacks.EarlyStopping(monitor='val_acc', patience=30, restore_best_weights=True)

        checkpoint = callbacks.ModelCheckpoint(args.task + ".h5", monitor='val_acc', save_best_only=True)

        history = model.fit(x=train_x, y=train_y, validation_data=validation_data, batch_size=batch_size, epochs=epochs, callbacks=[reduce_lr, earlyStopping, checkpoint])
        num_samples = 'all' if not train_samples else train_samples
        Models.saveHistory(history.history, "../trainingHistories/onCluster_{}_{}_{}_{}_{}_{}.pickle".format(args.task, num_samples, modelName, batch_size, input_shape, datetime.datetime.now()))
        
    elif args.task == 'bestFaceEndToEndFromCheckpoint':
            print("start: {}".format(datetime.datetime.now()))
            # Get all sample pathes
            train, test = splitDataSet(dataPath, 0.15, 42) # 1 to 1 Dataset
            print('Loaded {} training samples and {} validation samples'.format(len(train), len(test)))

            # Load samples to RAM
            batch_size = 64
            imageSize = (96, 96) # faces
            # imageSize = None # lips
            grayscale = False
            num_steps = 36 # 38 is max 
            one_hot = False
            train_samples= None 
            valid_samples= None  

            #Load model from CHECKPOINT
            model = load_model('bestFaceEndToEnd.h5')
            modelName = "TheFuckingLastModel"

            (train_x, train_y), validation_data = genData(train, 
                                                        test,train_samples=train_samples, 
                                                        valid_samples= valid_samples, 
                                                        batch_size = batch_size, 
                                                        imageSize = imageSize,
                                                        grayscale = grayscale, 
                                                        num_steps=num_steps, 
                                                        one_hot=one_hot,
                                                        print_freq=1000
                                                        )
            input_shape = train_x[0].shape
            print(input_shape)
            epochs = 200
            baseModel = "MobileNet"
            # model, modelName = Models.buildTimedistributed(baseModel, 
            #                                             input_shape=input_shape, 
            #                                             #lstm_dims = 128, 
            #                                             #num_lstm_layers= 4, 
            #                                             #num_dense_layers=0,
            #                                             #dense_dims=512,
            #                                             #base_model_weights='../models/mobileNetBaselineWeights.h5'
            #                                             )



            # define callbacks
            reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.1,
                                patience=10, min_lr=0.001, cooldown=2)
            earlyStopping = callbacks.EarlyStopping(monitor='val_acc', patience=30, restore_best_weights=True)

            filepath = "saved-e2ef-model-{epoch:02d}-{val_acc:.2f}.h5"
            checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_acc', save_best_only=False)

            history = model.fit(x=train_x, y=train_y, validation_data=validation_data, batch_size=batch_size, epochs=epochs, callbacks=[reduce_lr, earlyStopping, checkpoint])
            num_samples = 'all' if not train_samples else train_samples
            Models.saveHistory(history.history, "../trainingHistories/onCluster_{}_{}_{}_{}_{}_{}.pickle".format(args.task, num_samples, modelName, batch_size, input_shape, datetime.datetime.now()))
            