import collections
import csv
import time
import numpy as np
import tensorflow as tf

keras = tf.keras


def get_lr_metric(optimizer):
    """
        Get learning rate from optimizer

        Args:
            optimizer (optimizer): Keras optimizer
        Returns:
            lr (float): Learning rate of the optimizer
    """

    def lr(y_true, y_pred):
        return optimizer.learning_rate

    return lr


class CSVLogger(keras.callbacks.Callback):
    """Callback that streams epoch results to a CSV file.

    Supports all values that can be represented as a string,
    including 1D iterables such as `np.ndarray`.

    Example:

    ```python
    csv_logger = CSVLogger('training.log')
    model.fit(X_train, Y_train, callbacks=[csv_logger])
    ```

    Args:
        filename: Filename of the CSV file, e.g. `'run/log.csv'`.
        separator: String used to separate elements in the CSV file.
        append: Boolean. True: append if file exists (useful for continuing
            training). False: overwrite existing file.
    """

    def __init__(self, filename, separator=",", append=False):
        self.sep = separator
        self.filename = tf.compat.path_to_str(filename)
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        super().__init__()

    def on_train_begin(self, logs=None):
        """
        Initializes the training process, setting up the CSV file for logging.

        This method is called at the beginning of training. It opens the CSV file for
        logging training metrics, either in append or write mode based on the
        configuration.

        Args:
            logs (dict, optional): Can be used to pass training logs.
        """
        if self.append:
            if tf.io.gfile.exists(self.filename):
                with tf.io.gfile.GFile(self.filename, "r") as f:
                    self.append_header = not bool(len(f.readline()))
            mode = "a"
        else:
            mode = "w"
        self.csv_file = tf.io.gfile.GFile(self.filename, mode)

    def on_epoch_begin(self, epoch, logs=None):
        """
        Records the start time of the current epoch.

        This method is called at the beginning of each epoch and is used to record the
        start time of the epoch to calculate its duration later.

        Args:
            epoch (int): The index of the epoch that is beginning.
            logs (dict, optional): Can be used to pass epoch logs.
        """
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        """
        Logs the metrics at the end of an epoch to a CSV file.

        This method is called at the end of each epoch. It calculates the duration of
        the epoch,
        processes the log data, and writes the log data to a CSV file.

        Args:
            epoch (int): The index of the epoch that has ended.
            logs (dict, optional): A dictionary of metrics and their values at the end
            of the epoch.
        """
        logs = logs or {}

        epoch_time_end = time.time()
        duration = epoch_time_end - self.epoch_time_start

        logs["duration"] = duration

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, str):
                return k
            elif (
                    isinstance(k, collections.abc.Iterable)
                    and not is_zero_dim_ndarray
            ):
                return f"\"[{', '.join(map(str, k))}]\""
            else:
                return k

        if self.keys is None:
            self.keys = sorted(logs.keys())
            # When validation_freq > 1, `val_` keys are not in first epoch logs
            # Add the `val_` keys so that its part of the fieldnames of writer.
            val_keys_found = False
            for key in self.keys:
                if key.startswith("val_"):
                    val_keys_found = True
                    break
            if not val_keys_found:
                self.keys.extend(["val_" + k for k in self.keys])

        if not self.writer:

            class CustomDialect(csv.excel):
                delimiter = self.sep

            fieldnames = ["epoch"] + self.keys

            self.writer = csv.DictWriter(
                self.csv_file, fieldnames=fieldnames, dialect=CustomDialect
            )
            if self.append_header:
                self.writer.writeheader()

        row_dict = collections.OrderedDict({"epoch": epoch})
        row_dict.update(
            (key, handle_value(logs.get(key, "NA"))) for key in self.keys
        )
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        """
        Finalizes the training process by closing the CSV file.

        This method is called at the end of training. It closes the CSV file that was
        used
        for logging training metrics.

        Args:
            logs (dict, optional): Currently not used, but can be used to pass training
            logs.
        """
        self.csv_file.close()
        self.writer = None
