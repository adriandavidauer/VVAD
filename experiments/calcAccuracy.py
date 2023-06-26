"""Script to calculate the Accuracy for a given set of samples(test set)"""
# System imports
import argparse

# 3rd Party imports
from vvadlrs3.utils import kerasUtils
from vvadlrs3 import pretrained_models

# local imports

# end file header

__author__ = 'Adrian Lubitz'

# TODO: this is also implemented in the experiments notebook
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "path", help="path to a folder holding pickle files", type=str)

    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("-f", "--feature_type", help="Type for usage of a pretrained model.",
                        choices=["faceImage", "lipImage", "faceFeatures", "lipFeatures"], type=str)

    action.add_argument("-m", "--model_path",
                        help="Path to a model.", type=str)

    args = parser.parse_args()

    if args.model_path:
        model_path = args.model_path
    elif args.feature_type:
        if args.feature_type == "faceImage":
            model_path = pretrained_models.getFaceImageModelPath()
        elif args.feature_type == "lipImage":
            model_path = pretrained_models.getLipImageModelPath()
        elif args.feature_type == "faceFeatures":
            model_path = pretrained_models.getFaceFeatureModelPath()
        elif args.feature_type == "lipFeatures":
            model_path = pretrained_models.getLipFeatureModelPath()

    acc, (mae, maeStd), (mse, mseStd),  errors = kerasUtils.testModel(
        model_path, args.path)
    print("Accuracy for {} is {}".format(model_path, acc))
    print("MAE for {} is {} with std {}".format(
        model_path, mae, maeStd))
