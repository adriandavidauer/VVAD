"""A Script to convert samples into images"""
import argparse
import glob
import os
import pickle
import random
import sys

from PIL import Image

# 3rd Party imports

# local imports

# end file header

__author__ = 'Adrian Auer'
__copyright__ = 'Copyright (c)2017, Blackout Technologies'


def convert_samples_to_images(input_path, output_path='generatedImages', num=20):
    """
    Takes a path to a folder of samples and converts a given number of randomly picked
    samples(pickle files)
    and converts them to png images and saves them in the given output path

    :param input_path: Path to the samples
    :type input_path: String
    :param output_path: path where you want to save the generated images
    :type output_path: String
    :param num: number of samples to convert
    :type num: int
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    random_samples_from_list = random.sample(glob.glob(os.path.join(input_path,
                                                                    "*.pickle")),
                                             k=num)
    print("samples from list are: ", random_samples_from_list)

    for filepath in random_samples_from_list:
        with open(filepath, 'rb') as pickle_file:
            sample = pickle.load(pickle_file)
            data = sample['data'][0]
            img = Image.fromarray(data, 'RGB')
            b, g, r = img.split()
            img = Image.merge("RGB", (r, g, b))
            out_file_name = os.path.join(output_path,
                                         os.path.basename(filepath).strip('.pickle') +
                                         '.png')
            print("Generated image file is: ", out_file_name)
            img.save(out_file_name)
            # img.show()
            # print(sample)


def parse_args(args):
    fct_parser = argparse.ArgumentParser()
    fct_parser.add_argument("input_path", help='Path to the samples', type=str)
    fct_parser.add_argument("-o", "--output_path",
                            help="path where you want to save the "
                                 "generated images.", type=str)
    fct_parser.add_argument("-n", "--num", help="number of samples to convert",
                            type=int)
    return fct_parser.parse_args(args)


if __name__ == "__main__":   # pragma: no cover

    parser = parse_args(sys.argv[1:])

    convert_samples_to_images(parser.input_path, parser.output_path, parser.num)
