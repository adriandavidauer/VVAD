import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb

(x_train, y_train),(x_test, y_test) = imdb.load_data(num_words=n_unique_words)


###To fit the data into any neural network, we need to convert the data into sequence
# matrices. For this, we are using the pad_sequence module from keras.preprocessing.

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
y_train = np.array(y_train)
y_test = np.array(y_test)


####MODEL#######
model = Sequential()
model.add(Embedding(n_unique_words, 128, input_length=maxlen))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


'''''TEST MODEL 2'''
embedding_vector_features = 40
model = Sequential()
model.add(Embedding(vocab_size,embedding_vector_features,input_length=length))
model.add(Bidirectional(LSTM(100)))
model.add(Dense(1,activation=’sigmoid’))
model.compile(loss=’binary_crossentropy’,optimizer=’adam’,metrics=[‘accuracy’])
print(model.summary())

model = Sequential()
model.add(Embedding(n_unique_words, 128, input_length=maxlen))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
##
model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True),
                             input_shape=(5, 10)))
model.add(Bidirectional(LSTM(10)))
model.add(Dense(5))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# With custom backward layer
model = Sequential()
forward_layer = LSTM(10, return_sequences=True)
backward_layer = LSTM(10, activation='relu', return_sequences=True,
                      go_backwards=True)
model.add(Bidirectional(forward_layer, backward_layer=backward_layer,
                        input_shape=(5, 10)))
model.add(Dense(5))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
##
class LSTM_Model():

    """
    Bidirectional LSTM, with two hidden LSTM layers
		(forwards and backwards), both containing 93 one-cell
		memory blocks of one cell each (BLSTM)

		All nets contained an input layer of size 26 (one for
		each MFCC coefficient), and an output layer of size 61
		(one for each phoneme).

		The LSTM blocks had the
		following activation functions: logistic sigmoids in the
		range [K2, 2] for the input and output squashing
		functions of the cell, and in the range [0, 1] for the
		gates.

		For the output layers, we used the cross entropy error
		function and the softmax activation function, as is standard
		for 1 of K classification


        Dense/fully connected layer: A linear operation on the layer’s input vector
    """

    model = Sequential()
    # Input layer?
    model.add(Input(shape=(16,)))
    # Two fully connected layers
    model.add(Dense())
    #model.add(Bidirectional(LSTM(64, activation='relu')))
    forward_layer = LSTM(10, return_sequences=True)
    backward_layer = LSTM(1, activation='relu', return_sequences=True,
                          go_backwards=True)
    smallTestCNN.add(BatchNormalization(axis=chanDim))
    smallTestCNN.add(BatchNormalization(axis=chanDim))
    model.add(Bidirectional(forward_layer, backward_layer=backward_layer,
                            input_shape=(5, 10)))
    # Batch normalization between the LSTM layers

    model.add(Activation('softmax'))
    # or localModel.add(Dense(1, activation="softmax"))?


    #maybe like that
    """
    # Create our convnet with (112, 112, 3) input shape
    convnet = build_convnet(shape[1:])
    
    CREATE Model with dense and BLSTM earlier and apply it to time distributed
    Output comes later
    
    # then create our final model
    model = keras.Sequential()
    # add the convnet with (5, 112, 112, 3) shape
    model.add(TimeDistributed(convnet, input_shape=shape))
    
    """
