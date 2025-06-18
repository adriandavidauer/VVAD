from typing import Tuple, Optional

import tensorflow as tf
from keras.layers import Bidirectional, Dense, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.losses import BinaryCrossentropy
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

    def __init__(self):
        pass

    @staticmethod
    def build_land_lstm(input_shape: Tuple[int, int, int], num_td_dense_layers: int = 1,
                        num_blstm_layers: int = 2, dense_dims: int = 96,
                        hp: Optional = None) -> Sequential:
        """
        Builds a Land-LSTM model, optionally using a tuner for hyperparameter
        optimization.

        Args:
            input_shape (tuple): Input shape of the data.
            num_td_dense_layers (int): Number of time distributed dense layers.
            num_blstm_layers (int): Number of bidirectional LSTM layers.
            dense_dims (int, optional): Number of dense dimensions for the LSTM model,
                default is 96.
            hp (optional): Hyperparameter input for Keras tuner. If provided, dense
                layer units are tuned.

        Returns:
            Sequential: Returns the created Land-LSTM model.
        """
        land_lstm_model = Sequential(name="LandLSTM")

        # Flatten at first (providing original feature data)
        land_lstm_model.add(TimeDistributed(
            Flatten(input_shape=(input_shape[-2], input_shape[-1]))))

        if hp is not None:
            # If hp is provided, build the model using hyperparameter tuning
            land_lstm_model.add(
                TimeDistributed(
                    Dense(
                        units=hp.Int('units', min_value=32, max_value=512, step=32),
                        activation='relu'),
                    input_shape=input_shape))
            for _ in range(num_td_dense_layers):
                land_lstm_model.add(
                    TimeDistributed(
                        Dense(
                            units=hp.Int('units', min_value=32, max_value=512, step=32),
                            activation='relu')))
        else:
            # If hp is not provided, build the model using fixed dense dimensions
            land_lstm_model.add(TimeDistributed(Dense(dense_dims, activation='relu'),
                                                input_shape=input_shape))
            for _ in range(num_td_dense_layers):
                land_lstm_model.add(
                    TimeDistributed(Dense(dense_dims, activation='relu')))

        for j in range(num_blstm_layers):
            land_lstm_model.add(Bidirectional(
                layer=tf.keras.layers.LSTM(93, activation='relu',
                                           return_sequences=True)))
            if j < num_blstm_layers - 1:
                land_lstm_model.add(BatchNormalization())

            # Flatten again to reduce dimensionality
        land_lstm_model.add(Flatten())

        # Opposite to paper: Use 'sigmoid' activation for binary classification
        land_lstm_model.add(Dense(1, activation='sigmoid'))

        if hp is not None:
            loss = BinaryCrossentropy()
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            land_lstm_model.compile(optimizer=optimizer, loss=loss,
                                    metrics=[
                                        tf.keras.metrics.BinaryAccuracy(threshold=0.5),
                                        'TrueNegatives',
                                        'TruePositives',
                                        'FalseNegatives', 'FalsePositives'])

        return land_lstm_model
