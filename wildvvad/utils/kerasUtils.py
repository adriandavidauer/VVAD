from sklearn.model_selection import train_test_split


class kerasUtils:
    def __init__(self):
        pass

    @staticmethod
    def train_test_split(dataset: list) -> (list, list, list, list):
        """
        Uses the dataset to split into x_train, x_test, y_train, and y_test.

        Args:
        dataset (list): Dataset as list of dict.
        """

        X = []
        y = []
        for i in range(len(dataset)):
            X.extend(dataset[i]["sample"])
            y.append(1 if dataset[i]["label"] else 0)
            print(f"Current ds y is {y}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        return X_train, X_test, y_train, y_test
