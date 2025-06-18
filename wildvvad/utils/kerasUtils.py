from sklearn.model_selection import train_test_split


class kerasUtils:
    def __init__(self):
        pass

    @staticmethod
    def train_test_split(dataset: list, test_size: float = 0.2,
                         random_state: int = 42) -> (list, list, list, list):
        """
        Uses the dataset to split into x_train, x_test, y_train, and y_test.

        Args:
            dataset(list): Dataset as list of dict.
            test_size(float): fraction of the dataset reserved for testing
            random_state(int): Random state
        Returns:
            X_train, X_test, y_train, y_test(list, list, list, list): Split dataset
        """

        X = []
        y = []
        for i in range(len(dataset)):
            X.extend(dataset[i]["sample"])
            y.append(1 if dataset[i]["label"] else 0)
            print(f"Current ds y is {y}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size, random_state
        )

        return X_train, X_test, y_train, y_test

    def saveHistory(history, path):
        """
        Save a model's training history as a json file in given path

        Args:
            history (df) : Pandas dataframe containing training history of a model
            path (string): path to save the history to
        """
        with open(path, mode='w') as histFile:
            history.to_json(histFile)
