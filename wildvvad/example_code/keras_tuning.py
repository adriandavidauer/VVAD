# import
import os

import keras
from kerastuner.tuners import RandomSearch
import numpy as np
import tensorflow as tf

from wildvvad.utils import helper_functions
from wildvvad.utils.model import LAND_LSTM_Model

# Step 1: Define your model architecture
# From import

# Step 2: Get and process your data
y_train = np.array([0, 0, 0, 1, 1, 1])
x_train = np.array([0, 0, 0, 1, 1, 1])
x_test = np.array([0, 0, 0, 1, 1, 1])
y_test = np.array([0, 0, 0, 1, 1, 1])
x_val = np.array([0, 0, 0, 1, 1, 1])
y_val = np.array([0, 0, 0, 1, 1, 1])

# Step 3: Define the search space for hyperparameters
tuner = RandomSearch(
    LAND_LSTM_Model.build_land_lstm_tuner,
    objective='val_binary_accuracy',
    max_trials=10,  # Number of hyperparameter combinations to try
    executions_per_trial=1,  # Number of models to train per trial
    directory='tuner_test',  # Directory to save the tuning logs and checkpoints
    project_name='WildVVAD_tuner'  # Name of the tuning project
)

# Step 4: Configure the tuner
tuner.search_space_summary()

callbacks_array = []
callbacks_array.append(tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(".",
                          "checkpoints/tuner_test-{epoch:02d}.hdf5"),
    verbose=1,
    save_weights_only=False,
    save_freq=10
))

callbacks_array.append(tf.keras.callbacks.TensorBoard(
    log_dir=os.path.join(".", 'tensorboard_logs'),
    update_freq='epoch'
))
callbacks_array.append(
    keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', patience=15))

callbacks_array.append(
    helper_functions.CSVLogger(
        filename=os.path.join(".", 'tuner_outputs_csv.log')))
callbacks_array.append(
    tf.keras.callbacks.TensorBoard('./tuner_test/WildVVAD_tuner'))

# Step 5: Search for the best hyperparameters
tuner.search(x_train, y_train, epochs=60, validation_data=(x_val, y_val))

# Step 6: Save the best model and its corresponding hyperparameters
best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
print(best_hyperparameters.values)

try:
    with open('best_hyperparameters.txt', 'w') as f:
        f.write(best_hyperparameters.get_config_as_str())
except Exception as e:
    print("Saving hyperparameters did not work because of ", e)

best_model.build(best_hyperparameters, input_shape=((None,) + x_train.shape[1:]))

best_model.save('best_model.h5')

# Evaluate the best model
loss, accuracy = best_model.evaluate(x_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
