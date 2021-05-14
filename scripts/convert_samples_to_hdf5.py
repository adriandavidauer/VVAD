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

    parser.add_argument(
        "--shape", help="For the image types a shape can be given to which the image will be resized", type=int, nargs=2)
    args = parser.parse_args()
    input_folder = Path(args.input_folder)
    test_set = input_folder / 'testSet'

    train_samples = list(input_folder.glob('*/*.pickle'))
    test_samples = list(test_set.glob('*/*.pickle'))

    # Get shape of a sample
    if args.shape:
        resize_shape = tuple(args.shape)
    else:
        resize_shape = args.shape
    print(f'resize_shape: {resize_shape}')
    _sample = FeatureizedSample()
    _sample.load(train_samples[0])
    sample_shape = _sample.getData(resize_shape).shape

    print(f'sample_shape: {sample_shape}')

    x_chunk_shape = (args.chunk_resolution, *sample_shape)
    y_chunk_shape = (args.chunk_resolution,)

    out_h5 = h5py.File(args.output_file, "w")
    x_train_ds = out_h5.create_dataset(shape=(len(train_samples), *sample_shape), dtype=np.uint8,
                                       name="x_train", chunks=x_chunk_shape, compression="gzip", compression_opts=9)
    y_train_ds = out_h5.create_dataset(shape=(
        len(train_samples),), name="y_train", dtype=np.uint8, chunks=y_chunk_shape, compression="gzip", compression_opts=9)

    x_test_ds = out_h5.create_dataset(shape=(len(test_samples), *sample_shape), dtype=np.uint8,
                                      name="x_test", chunks=x_chunk_shape, compression="gzip", compression_opts=9)
    y_test_ds = out_h5.create_dataset(shape=(
        len(test_samples),), name="y_test", dtype=np.uint8, chunks=y_chunk_shape, compression="gzip", compression_opts=9)

    # Loop for the training samples
    print('converting training samples')
    for i, sample in enumerate(tqdm(train_samples)):
        s = FeatureizedSample()
        s.load(sample)
        data = s.getData(resize_shape)  # The np.array for that sample
        label = s.getLabel()  # The label as int for that sample

        x_train_ds[i] = data
        y_train_ds[i] = label

    # Loop for the test samples
    print('converting test samples')
    for i, sample in enumerate(tqdm(test_samples)):
        s = FeatureizedSample()
        s.load(sample)
        data = s.getData(resize_shape)  # The np.array for that sample
        label = s.getLabel()  # The label as int for that sample

        x_test_ds[i] = data
        y_test_ds[i] = label

    out_h5.close()
