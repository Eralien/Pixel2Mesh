import numpy as np 

LIST_PREFIX = './pixel2mesh/utils/'
MATCH_WORDS = 'data/'
REPLACE_WORDS = '/media/eralien/ReservoirLakeBed/Pixel2Mesh/'

def new_list_gen(from_list_path, to_list_path, read_num=None):
    with open(to_list_path, 'w') as f1:
        with open(from_list_path, 'r') as f2:
            i = 0
            for line in f2:
                # print(line)
                new_line = line.replace(MATCH_WORDS, REPLACE_WORDS, 1)
                # print(new_line)
                f1.write(new_line)
                
                if read_num!=None: 
                    i = i+1
                    if i>=read_num: break
            

if __name__ == "__main__":
    FROM_LIST = 'train_list.txt'
    TO_LIST = 'train_list_new.txt'
    new_list_gen(LIST_PREFIX+FROM_LIST, LIST_PREFIX+TO_LIST)
    pass