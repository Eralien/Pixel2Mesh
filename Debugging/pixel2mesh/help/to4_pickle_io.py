from __future__ import print_function
import numpy as np
import os
import cPickle as pickle
import cv2

def dat_gen(pkl_path, remove_origin=False):
    if remove_origin:
        try:
            os.remove(pkl_path)
        except OSError:
            pass

    for key, value in la_dict.iteritems():
        try:
            with open(pkl_path, 'rb') as pkl_file:
                pkl_dict = pickle.load(pkl_file)
        except IOError:
            pkl_dict = {}
        else:
            pass

        with open(pkl_path, 'wb') as pkl_file:
            pkl_dict.update({key: value})
            pickle.dump(pkl_dict, pkl_file)

        print('Iter')
        for pkl_key, pkl_value in pkl_dict.iteritems():
            print(str(pkl_key) + ':' + str(pkl_value))


def dat_read(pkl_path):
    with open(pkl_path, 'rb') as pkl_file:
        pkl_dict = pickle.load(pkl_file)
    return pkl_dict
            



if __name__ == "__main__":
    print(os.getcwd())

    # add a pickle dictionary to a file
    la_dict = {'A': 1, 'B': [[2, 2], [3, -1]],
               'C': {1: 1, '2': 2}, 'D': 'string bla'}

    # pkl_path = './Debugging/pkl_test.dat'
    # pkl_path = '/media/eralien/ReservoirLakeBed1/Pixel2Mesh/ShapeNetTrain/04256520_1a4a8592046253ab5ff61a3a2a0e2484_00.dat'
    pkl_path = './Debugging/pixel2mesh/help/initGraph.dat'
    pkl_dict = dat_read(pkl_path)
    img = pkl_dict[0].astype('float32')/255.0
    label = pkl_dict[1]
    
    pass