import keras
from keras.layers import Dropout, Bidirectional, Activation, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential


class LAND_LSTM_Model:
    """
        Build Bidirectional LSTM Model based on the WildVVAD description.
            Each video sample is mapped onto a sequence of vectors,
            where each vector contains the frontal 3D coordinates of the 68 landmarks.
            This sequence is then fed into a bidirectional LSTM, [16], in which fully
            connected layers share the parameters across time. We employ
            ReLu activations and add batch normalization layers [17]
            between bidirectional LSTM layers.
    """

    @staticmethod
    def build_land_lstm(input_shape, num_td_dense_layers=2,
                        num_blstm_layers=2, dense_dims=512) -> (Sequential, str):
        """Building model

        Args:
            input_shape(tuple): input shape of the data
            num_td_dense_layers(int): number of time distributed dense layers
            num_blstm_layers(int): number of bidirectional lstm layers
            dense_dims (int): number of dense dimensions for the lstm model, default = 512

        Returns:
            localModel (Sequential()): Returns created Land-LSTM model
            modelName (String): Model name with all layer and dimension information
        """
        land_lstm_model = Sequential()
        # ToDo: Input layer needed?

        # Two fully connected layers
        # ToDo unclear about input shape like in
        #   add the convnet with (5, 112, 112, 3) shape
        #   model.add(TimeDistributed(convnet, input_shape=shape))
        for i in range(num_td_dense_layers - 1):
            land_lstm_model.add(TimeDistributed(Dense(dense_dims),
                                                input_shape=input_shape))

        # model.add(Bidirectional(LSTM(64, activation='relu')))
        forward_layer = LSTM(10, activation='relu', return_sequences=True)
        backward_layer = LSTM(1, activation='relu', return_sequences=True,
                              go_backwards=True)
        for j in range(num_blstm_layers - 1):
            land_lstm_model.add(Bidirectional(layer=forward_layer,
                                              backward_layer=backward_layer,
                                              input_shape=(5, 10)))

            # Batch normalization between the LSTM layers
            if j < num_blstm_layers:
                land_lstm_model.add(BatchNormalization())
        land_lstm_model.add(Dropout(0.5))

        # ToDo: model.add(Dense(1, activation="softmax"))?
        # Alternative:
        land_lstm_model.add(Activation('softmax'))
        land_lstm_model.compile(loss="binary_crossentropy",
                                optimizer='sgd', metrics=["accuracy"])

        model_name = 'Land-LSTM_' + \
                     str(num_td_dense_layers) + '_' + str(num_blstm_layers)
        return land_lstm_model, model_name


if __name__ == "__main__":
    pass
    # model = LAND_LSTM_Model()
    # lstm, name = model.build_land_lstm(input_shape=(200,200))
    # print(name)
