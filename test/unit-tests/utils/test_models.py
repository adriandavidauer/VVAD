import unittest
from unittest.mock import Mock

import keras

from parameterized import parameterized

from wildvvad.model import LAND_LSTM_Model


class TestModelUtils(unittest.TestCase):
    """
        Test the models regarding WildVVAD.
    """

    def setUp(self):
        """
            Unit test initial setup before every test run.
            Defining the shape of the input sample
        """

        self.input_shape = (68, 38, 3)
        self.num_td_dense_layers = 2
        self.num_blstm_layers = 3
        self.hp = Mock()  # Create a mock object for hp
        # Mock the Int method to return a fixed value
        self.hp.Int.return_value = 128  # Example fixed value

    def test_model_creation(self):
        """
            Unit test on successful creation of the given model
        """

        model = LAND_LSTM_Model.create_land_lstm(self.input_shape)
        self.assertIsInstance(model, keras.Sequential)
        self.assertEqual(model.name, "LandLSTM")

    @parameterized.expand([
        (0, 0),
        (1, 1),
        (2, 3),
        (10, 10),
        (0, 10),
        (10, 0),
        (1, 0),
        (0, 1),
        (-1, 3),  # Negative TD layers
        (2, -2),  # Negative BLSTM layers
        (-1, -1),  # Both negative
        # Add more combinations as needed
    ])
    def test_layer_addition(self, num_td_dense_layers, num_blstm_layers):
        """
            Verify that the provided number of time-distributed dense layers and
            bidirectional LSTM layers are added to the model.

            Args:
                num_td_dense_layers (int): Number of time distributed dense layers
                num_blstm_layers (int): Number of bidirectional lstm layers
        """

        dense_dims = 96
        if num_td_dense_layers < 0 or num_blstm_layers < 0:
            with self.assertRaises(ValueError):
               LAND_LSTM_Model.create_land_lstm(self.input_shape,
                                                 num_td_dense_layers,
                                                 num_blstm_layers,
                                                 dense_dims)
        else:
            model = LAND_LSTM_Model.create_land_lstm(self.input_shape,
                                                     num_td_dense_layers,
                                                     num_blstm_layers,
                                                     dense_dims)

            # Check time-distributed dense layers
            td_dense_layers = [layer for layer in model.layers if
                               isinstance(layer, keras.layers.TimeDistributed)]
            self.assertEqual(len(td_dense_layers), num_td_dense_layers + 2,
                             "Incorrect number of TimeDistributed layers")

            # Check bidirectional LSTM layers
            blstm_layers = [layer for layer in model.layers if
                            isinstance(layer, keras.layers.Bidirectional)]
            self.assertEqual(len(blstm_layers), num_blstm_layers,
                             "Incorrect number of Bidirectional LSTM layers")

    def test_layer_connections(self):
        """
        For each layer, we check that its input comes from the output of the previous layer.
        This ensures that the layers are properly connected in sequence as expected in the model architecture.
        """
        model = LAND_LSTM_Model.create_land_lstm(self.input_shape)
        model.build(input_shape=self.input_shape)
        layers = model.layers
        for i in range(1, len(layers)):
            self.assertIs(layers[i].input, layers[i - 1].output)

    def test_output_layer(self):
        """
        Check that the output layer has the correct number of units and activation function for binary classification.
        """
        model = LAND_LSTM_Model.create_land_lstm(self.input_shape)
        output_layer = model.layers[-1]
        self.assertEqual(output_layer.units, 1)
        self.assertEqual(output_layer.activation, keras.activations.sigmoid)


if __name__ == '__main__':
    unittest.main(warnings='ignore')
