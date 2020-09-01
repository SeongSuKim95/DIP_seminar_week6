
import cv2
import numpy as np
import glob

def Video_creator():
    img_array = []

    for filename in glob.glob('C:/Users/user/PycharmProjects/DIP week6/Tracking Set 1/*.bmp'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter('project1.avi', cv2.VideoWriter_fourcc(*'DIVX'), 20, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

Video_creator()