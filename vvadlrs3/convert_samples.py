'''A Script to convert samples into images'''
# System imports
import glob, os
import argparse
import random 
import pickle
from PIL import Image

# 3rd Party imports

# local imports

# end file header

__author__      = 'Adrian Lubitz'
__copyright__   = 'Copyright (c)2017, Blackout Technologies'

def convert_samples(input_path, output_path='generatedImages', num=20):
    '''
    Takes a path to a folder of samples and converts a given number of randomly picked samples(pickle files) 
    and converts them to png images and saves them in the given output path

    :param input_path: Path to the samples
    :type input_path: String
    :param output_path: path where you want to save the generated images
    :type output_path: String
    :param num: number of samples to convert
    :type num: int
    '''
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #TODO for sample in input_path - convert and safe to output_path
    for filepath in random.choices(glob.glob(os.path.join( input_path, "*.pickle")), k=num):
        with open(filepath, 'rb') as pickle_file:
            sample = pickle.load(pickle_file)
            data = sample['data'][0]
            print(data.shape)
            img = Image.fromarray(data, 'RGB')
            b, g, r = img.split()
            img = Image.merge("RGB", (r, g, b))  
            out_file_name = os.path.join(output_path, os.path.basename( filepath).strip('.pickle') + '.png') 
            print(out_file_name)
            img.save(out_file_name)
            # img.show()
            # print(sample)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help='Path to the samples' , type=str)
    parser.add_argument("-o","--output_path", help="path where you want to save the generated images.", type=str)
    parser.add_argument("-n", "--num", help="number of samples to convert" , type=int)
    
    args = parser.parse_args()

    convert_samples(args.input_path, args.output_path, args.num)
