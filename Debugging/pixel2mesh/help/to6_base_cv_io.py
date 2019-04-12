import numpy as np 
import os
import cv2
import os

def show_img(img_path, name='img'):
    img_mat = cv2.imread(img_path,-1)
    cv2.imshow(name, img_mat)
    cv2.waitKey(0)
    return img_mat


if __name__ == "__main__":
    print(os.getcwd())

    img_path = './pixel2mesh/utils/examples/car.png'
    car_pic = show_img(img_path, name='car_pic')
    cv2.imwrite('./Debugging/pic/img.png', car_pic)