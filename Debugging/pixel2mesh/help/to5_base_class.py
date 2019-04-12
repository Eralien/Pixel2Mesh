import numpy as np 
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

    A = Class0(a=3, b=4)
    A.print_out()
    B = Class1(a=3, b=4)
    B.print_out()
    C = []
    C.append(A)
    C.append(B)