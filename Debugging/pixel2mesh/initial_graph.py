import numpy as np 
import os 
import sys
import matplotlib.pyplot as plt
import time

class InitGraph(object):
    def __init__(self, size=np.array([6,8]), name='init_graph'):
        self.dt = np.dtype([('name', np.unicode_, 16), ('position', np.float16, 2), ('coord', np.int16, 2)])
        self.init = np.empty(size, dtype=dt)
        self.index = 0
        pass
    
    def __call__(self, name, lanes, h_samples):
        self.name = name
        self.lanes = np.array(lanes, dtype=self.dt)
        self.h_samples = np.array(h_samples, dtype=self.dt)
        for name, pos, coor in zip(list_name, list_pos, list_coor):

        return self.write_in()

    def __del__(self):
        pass
    
    def __iter__(self):
        return self
    
    # def next(self):
    #     if self.index == 0:
    #         raise StopIteration
    #     self.index =  self.index - 1
    #     return self.output(self.index)

    def write_in(self):
        output = self.lanes + self.h_samples
        return output
        
    



if __name__ == "__main__":
    initGraph = InitGraph()
    list_name = ['a1', 'a2', 'a3']
    list_coor = range(3)
    list_pos = np.random.rand(3,2)
    # list_wrap = {'name':list_name, 'coor':list_coor, 'pos':list_pos}
    output = initGraph(list_name, list_pos, list_coor)
