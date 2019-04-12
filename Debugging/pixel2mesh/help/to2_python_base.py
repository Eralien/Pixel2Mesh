import numpy as np 
import os
import cPickle as pickle
import cv2
import os

class Class0(object):
    def __init__(self, **argv):
        self.init0 = []
        self.init1 = []
        self.init2 = {}
        self.a = argv['a']
        self.b = argv['b']
        
    def iter1(self):
        self.c = self.a**2
        self.iter2()
    
    def iter2(self):
        self.d = self.b**2

    def print_out(self):
        self.iter1()
        print('c = ' + str(self.c))
        print('d = ' + str(self.d))
        

class Class1(Class0):
    def __init__(self, **argv):
        super(Class1, self).__init__(**argv)
    
    def iter2(self):
        self.d = self.b**3
    

if __name__ == "__main__":
    print(os.getcwd())

    # img_path = './pixel2mesh/utils/examples/car.png'
    # car_pic = cv2.imread(img_path,-1)
    # cv2.imshow('car', car_pic)
    # cv2.waitKey(0)

    A = Class0(a=3, b=4)
    A.print_out()
    B = Class1(a=3, b=4)
    B.print_out()
    C = []
    C.append(A)
    C.append(B)



    # add a pickle dictionary to a file
    la_dict = {'A': 1, 'B': [[2, 2], [3, -1]],
               'C': {1: 1, '2': 2}, 'D': 'string bla'}

    pkl_path = './Debugging/pkl_test.dat'
    
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
            pkl_dict.update({key:value})
            pickle.dump(pkl_dict, pkl_file)

        print('Iter')
        for pkl_key, pkl_value in pkl_dict.iteritems():
            print(str(pkl_key) + ':' + str(pkl_value))

    pass
