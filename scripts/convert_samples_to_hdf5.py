'''
This script converts samples into hdf5 format
'''

# System imports
import argparse
from pathlib import Path

# 3rd party imports
from vvadlrs3.sample import FeatureizedSample
from tqdm import tqdm


# local imports

# end file header
__author__ = 'Adrian Lubitz'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_folder", help="Path the folder which holds the folders 'negativeSamples', 'positiveSamples' and 'testSet'", type=str)
    parser.add_argument(
        "output_file", help="Path the hdf5 file representing the dataset", type=str)
    args = parser.parse_args()
    input_folder = Path(args.input_folder)
    test_set = input_folder / 'testSet'

    # Loop for the training samples
    print('converting training samples')
    for sample in tqdm(list(input_folder.glob('*/*.pickle'))):
        s = FeatureizedSample()
        s.load(sample)
        data = s.getData()  # The np.array for that sample
        label = s.getLabel()  # The label as int for that sample
        # TODO: @matias: add hdf5 magic here

    # Loop for the test samples
    print('converting test samples')
    for sample in tqdm(list(test_set.glob('*/*.pickle'))):
        s = FeatureizedSample()
        s.load(sample)
        data = s.getData()  # The np.array for that sample
        label = s.getLabel()  # The label as int for that sample
        # TODO: @matias: add hdf5 magic here
