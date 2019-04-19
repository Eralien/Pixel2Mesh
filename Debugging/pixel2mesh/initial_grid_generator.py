import cv2
import json
import numpy as np
import os
from initial_graph import *
 
tusimple_path_base = '/media/eralien/ReservoirLakeBed/Pixel2Mesh/TuSimple/'

def img_save_path_check(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_path_fillin(binary_path, image_num):
    return binary_path + str(image_num) + '.png'

def dataset_gen(tusimple_path_base, train_flag=True, 
binary_path=None, instance_path=None,output_path=None):
    if train_flag:
        tusimple_path = tusimple_path_base + 'train_set/'
    else:
        tusimple_path = tusimple_path_base + 'test_set/'

    file = open(tusimple_path + 'label_data_0601.json','r')
    image_num = 0

    binary_path = tusimple_path + 'image_binary/' + str(image_num) + '.png'
    instance_path = tusimple_path + 'image_instance/' + str(image_num) + '.png'
    color_path = tusimple_path + 'image_color/' + str(image_num) + '.png'
    img_save_path_check(binary_path)
    img_save_path_check(instance_path)
    img_save_path_check(color_path)

    for line in file.readlines():
        data = json.loads(line)
        image = cv2.imread(tusimple_path + data['raw_file'])
        binaryimage = np.zeros((image.shape[0], image.shape[1],1), np.uint8)
        instanceimage = binaryimage.copy()
        arr_width = data['lanes']
        arr_height = data['h_samples']
        width_num = len(arr_width)
        height_num = len(arr_height)
        for i in range(height_num):
            lane_hist = 20
            for j in range(width_num):
                if arr_width[j][i-1] > 0 and arr_width[j][i] > 0:
                    binaryimage[int(arr_height[i]),int(arr_width[j][i])] = 255
                    instanceimage[int(arr_height[i]),int(arr_width[j][i])] = lane_hist
                    if i>0:
                        pass
                lane_hist += 50
        
        # save_binary_path = save_path_fillin(binary_path, image_num)
        # save_instance_path = save_path_fillin(instance_path, image_num)
        # save_color_path = save_path_fillin(color_path, image_num)

        # cv2.imwrite(save_binary_path, binaryimage)
        # cv2.imwrite(save_instance_path, instanceimage)
        # cv2.imwrite(save_color_path, image)
        image_num += 1
        if image_num == 15:
            break
    file.close()

if __name__ == "__main__":
    initGraph = InitGraph()
    dataset_gen(tusimple_path_base)