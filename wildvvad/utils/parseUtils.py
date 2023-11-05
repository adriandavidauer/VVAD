import argparse


class parseUtils:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("option", help="what you want to do.", type=str,
                                 choices=[
                                     "download", "createDataset",
                                     "trainModel"
                                 ])
        self.parser.add_argument(
            "--debug", help="debug messages will be displayed", action='store_true')
