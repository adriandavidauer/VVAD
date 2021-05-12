'''
This script converts samples into hdf5 format
'''

# System imports
import argparse
from pathlib import Path

# 3rd party imports
from vvadlrs3.sample import FeatureizedSample
from tqdm import tqdm

import h5py
import numpy as np

# local imports

# end file header
__author__ = 'Adrian Lubitz'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_folder", help="Path the folder which holds the folders 'negativeSamples', 'positiveSamples' and 'testSet'", type=str)
    parser.add_argument(
        "output_file", help="Path the hdf5 file representing the dataset", type=str)

    parser.add_argument(
        "--chunk_resolution", help="How many samples to take for each chunk, affects compression performance", type=int, default=16)
    args = parser.parse_args()
    input_folder = Path(args.input_folder)
    test_set = input_folder / 'testSet'

    x_chunk_shape = (args.chunk_resolution, 38, 200, 200, 3)
    y_chunk_shape = (args.chunk_resolution,)

    train_samples = list(input_folder.glob('*/*.pickle'))
    test_samples = list(test_set.glob('*/*.pickle'))

    print(f'len(train_samples): {len(train_samples)}')
    print(f'len(test_samples): {len(test_samples)}')

    out_h5 = h5py.File(args.output_file, "w")
    x_train_ds = out_h5.create_dataset(shape=(len(train_samples), 38, 200, 200, 3), dtype=np.uint8,
                                       name="x_train", chunks=x_chunk_shape, compression="gzip", compression_opts=9)
    y_train_ds = out_h5.create_dataset(shape=(
        len(train_samples),), name="y_train", dtype=np.uint8, chunks=y_chunk_shape, compression="gzip", compression_opts=9)

    x_test_ds = out_h5.create_dataset(shape=(len(test_samples), 38, 200, 200, 3), dtype=np.uint8,
                                      name="x_test", chunks=x_chunk_shape, compression="gzip", compression_opts=9)
    y_test_ds = out_h5.create_dataset(shape=(
        len(test_samples),), name="y_test", dtype=np.uint8, chunks=y_chunk_shape, compression="gzip", compression_opts=9)

    # train_count = 0
    # test_count = 0

    # Loop for the training samples
    print('converting training samples')
    for i, sample in enumerate(tqdm(train_samples)):
        s = FeatureizedSample()
        s.load(sample)
        data = s.getData()  # The np.array for that sample
        label = s.getLabel()  # The label as int for that sample
        # TODO: @matias: add hdf5 magic here

        # x_train_ds.resize(train_count + 1, axis=0)
        # y_train_ds.resize(train_count + 1, axis=0)

        x_train_ds[i] = data
        y_train_ds[i] = label

        # train_count += 1

    # Loop for the test samples
    print('converting test samples')
    for i, sample in enumerate(tqdm(test_samples)):
        s = FeatureizedSample()
        s.load(sample)
        data = s.getData()  # The np.array for that sample
        label = s.getLabel()  # The label as int for that sample
        # TODO: @matias: add hdf5 magic here

        # x_test_ds.resize(test_count + 1, axis=0)
        # y_test_ds.resize(test_count + 1, axis=0)

        x_test_ds[i] = data
        y_test_ds[i] = label

        # test_count += 1

    out_h5.close()
