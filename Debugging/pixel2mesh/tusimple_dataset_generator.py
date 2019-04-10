import cv2
import json
import numpy as np
import os
from initial_graph import *

tusimple_path_base = '/media/eralien/ReservoirLakeBed1/Pixel2Mesh/TuSimple/'


def img_save_path_check(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_path_fillin(binary_path, image_num):
    return binary_path + str(image_num) + '.png'


def plot_line(height_num, width_num, arr_width, arr_height,
              binaryimage, instanceimage, image, lane_hist=None):
    
    if lane_hist==None: lane_hist = np.array([20, 20, 20])
    for i in range(height_num):
        hist = lane_hist.copy()
        for j in range(width_num):
            if arr_width[j][i-1] > 0 and arr_width[j][i] > 0:
                binaryimage[int(arr_height[i]), int(
                    arr_width[j][i])] = [255, 255, 255]
                instanceimage[int(arr_height[i]), int(
                    arr_width[j][i])] = hist

                if arr_width[j-1][i] > 0 and j > 0:
                    cv2.line(binaryimage, (int(arr_width[j-1][i]), int(arr_height[i])), (int(
                        arr_width[j][i]), int(arr_height[i])), (100, 100, 100), 1)
                    cv2.line(instanceimage, (int(arr_width[j-1][i]), int(arr_height[i])), (int(
                        arr_width[j][i]), int(arr_height[i])), (100, 100, 100), 1)
                    cv2.line(image, (int(arr_width[j-1][i]), int(arr_height[i])), (int(
                        arr_width[j][i]), int(arr_height[i])), (100, 100, 100), 1)

                if i > 0:
                    cv2.line(binaryimage, (int(arr_width[j][i-1]), int(arr_height[i-1])), (int(
                        arr_width[j][i]), int(arr_height[i])), np.array([255, 255, 255])/4, 10)
                    cv2.line(instanceimage, (int(arr_width[j][i-1]), int(arr_height[i-1])), (int(
                        arr_width[j][i]), int(arr_height[i])), hist, 10)
                    cv2.line(image, (int(arr_width[j][i-1]), int(arr_height[i-1])), (int(
                        arr_width[j][i]), int(arr_height[i])), hist, 10)
            hist += 50


def plot_circle(height_num, width_num, arr_width, arr_height,
                binaryimage, instanceimage, image,
                color_dist=None):
    if color_dist==None: color_dist = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0],
                                    [0, 1, 1], [1, 1, 0], [1, 0, 1]])*255
    for j in range(width_num):
        if j < 6:
            color_lanedot = color_dist[j, :].copy()
        for i in range(height_num):
            if arr_width[j][i] > 0:
                if j < 6:
                    cv2.circle(binaryimage, (int(arr_width[j][i]), int(
                        arr_height[i])), 7, color_lanedot, 2)
                    cv2.circle(instanceimage, (int(arr_width[j][i]), int(
                        arr_height[i])), 7, color_lanedot, 2)
                    cv2.circle(image, (int(arr_width[j][i]), int(
                        arr_height[i])), 7, color_lanedot, 2)
                else:
                    cv2.circle(binaryimage, (int(arr_width[j][i]), int(
                        arr_height[i])), 7, np.random.rand(1, 2)*255, 2)
                    cv2.circle(instanceimage, (int(arr_width[j][i]), int(
                        arr_height[i])), 7, np.random.rand(1, 2)*255, 2)
                    cv2.circle(image, (int(arr_width[j][i]), int(
                        arr_height[i])), 7, np.random.rand(1, 2)*255, 2)
            color_lanedot_ = np.vstack(
                (color_lanedot, np.ones([1, 3])*i/height_num*255))
            color_lanedot = np.amax(color_lanedot_, axis=0)


def dataset_gen(tusimple_path_base, initialGraph, train_flag=True,
                binary_path=None, instance_path=None, output_path=None, write=False):
    if train_flag:
        tusimple_path = tusimple_path_base + 'train_set/'
    else:
        tusimple_path = tusimple_path_base + 'test_set/'

    file = open(tusimple_path + 'label_data_0601.json', 'r')
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
        binaryimage = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
        instanceimage = binaryimage.copy()
        arr_width = data['lanes']
        arr_height = data['h_samples']
        width_num = len(arr_width)
        height_num = len(arr_height)
        initialGraph(arr_width, arr_height)

        plot_line(height_num, width_num, arr_width, arr_height,
                  binaryimage, instanceimage, image)

        plot_circle(height_num, width_num, arr_width, arr_height,
                    binaryimage, instanceimage, image)

        if write==True:
            save_binary_path = save_path_fillin(binary_path, image_num)
            save_instance_path = save_path_fillin(instance_path, image_num)
            save_color_path = save_path_fillin(color_path, image_num)
            cv2.imwrite(save_binary_path, binaryimage)
            cv2.imwrite(save_instance_path, instanceimage)
            cv2.imwrite(save_color_path, image)
        else:
            cv2.namedWindow("Image") 
            cv2.namedWindow("Binary") 
            cv2.namedWindow("Instance") 
            cv2.imshow("Image", image)
            cv2.imshow("Binary", binaryimage) 
            cv2.imshow("Instance", instanceimage) 
            cv2.waitKey(0)

        image_num += 1
        print("image_num: " + str(image_num))
        
        if image_num == 15:
            break
    file.close()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    initialGraph = InitGraph()
    dataset_gen(tusimple_path_base, initialGraph)
