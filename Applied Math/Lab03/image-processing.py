import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

def change_brightness(img_1d, bias):
    row, column, channel = img_1d.shape
    img_1d = img_1d.reshape(img_1d.shape[0] * img_1d.shape[1], img_1d.shape[2])
    img_1d=  img_1d.astype(np.uint16) + bias
    img_1d = np.clip(img_1d,0,255)
    img_1d = img_1d.reshape((row, column, channel))
    return img_1d

def change_contrast(img_1d, alpha):
    row, column, channel = img_1d.shape
    img_1d = img_1d.reshape(img_1d.shape[0] * img_1d.shape[1], img_1d.shape[2])
    img_1d=  img_1d.astype(np.uint16) * alpha
    img_1d = np.clip(img_1d,0,255)
    img_1d = img_1d.reshape((row, column, channel))
    return img_1d

def change_grayscale(img_1d, way = 'weight'):
    row, column, channel = img_1d.shape
    img_1d = img_1d.reshape(img_1d.shape[0] * img_1d.shape[1], img_1d.shape[2])
    if way == 'average':
        new_img_1d = np.zeros((img_1d.shape[0], 1))
        for i in range (new_img_1d.shape[0]):
            new_img_1d[i]+= np.mean(img_1d[i])
        new_img_1d = new_img_1d.reshape((row, column, 1))
        new_img_1d = np.concatenate((new_img_1d, new_img_1d, new_img_1d), -1)
        return new_img_1d
    if way == 'weight':
        r, g, b = img_1d[:, 0], img_1d[:, 1], img_1d[:, 2]
        new_img_1d = 0.2989 * r + 0.5870 * g + 0.1140 * b
        new_img_1d = new_img_1d.reshape(row, column, 1)
        new_img_1d = np.dstack((new_img_1d,new_img_1d,new_img_1d))
        return new_img_1d


def change_reflection(img_1d, way):
    if way == 'horizontal':
        new_img_1d = img_1d[0][::-1]
        for i in range(1,img_1d.shape[0]):
            temp = img_1d[i][::-1]
            new_img_1d = np.concatenate((new_img_1d, temp), axis=0)
        new_img_1d = new_img_1d.reshape((img_1d.shape[0], img_1d.shape[1], img_1d.shape[2]))
        return new_img_1d
    if way == 'vertical':
        new_img_1d = img_1d[::-1][0]
        for i in range(1, img_1d.shape[1]):
            temp = img_1d[::-1][i]
            new_img_1d = np.concatenate((new_img_1d, temp), axis=0)
        new_img_1d = new_img_1d.reshape((img_1d.shape[0], img_1d.shape[1], img_1d.shape[2]))
        return new_img_1d

def change_concatenate(img_1d_1, img_1d_2, alpha1):
    img_1d_1 = change_grayscale(img_1d_1)
    img_1d_2 = change_grayscale(img_1d_2)
    new_img_1d = alpha1 * img_1d_1 + (1-alpha1) * img_1d_2
    return new_img_1d


def change_blur(img_2d, kernel_size = 3):
    if kernel_size % 2 == 0:
        print('Kernel size must be odd \n')
        return img_2d
    padding = (kernel_size - 1 )/ 2
    padding = int(padding)
    new_img_2d = np.zeros((img_2d.shape[0], img_2d.shape[1], img_2d.shape[2]))
    for i in range(img_2d.shape[0]):
        for j in range(img_2d.shape[1]):
            kernel = img_2d[np.clip(i-padding,0,img_2d.shape[0]):np.clip(i+padding + 1,0,img_2d.shape[0]),np.clip(j-padding,0,img_2d.shape[1]):np.clip(j+padding + 1,0,img_2d.shape[1]),:]
            new_img_2d[i][j][0] = np.mean(kernel[:, :, 0])
            new_img_2d[i][j][1] = np.mean(kernel[:, :, 1])
            new_img_2d[i][j][2] = np.mean(kernel[:, :, 2])
    return new_img_2d

if __name__ == '__main__':
    #change input
    img_2d_1 = Image.open('img.jpg')
    img_2d_1 = np.asarray(img_2d_1)
    img_2d_2 = Image.open('horizontal.png')
    img_2d_2 = np.asarray(img_2d_2)
    #output
    img_2d = change_blur(img_2d_1,3)
    img_2d = img_2d.astype(np.uint8)
    plt.imshow(img_2d)
    plt.show()
    im = Image.fromarray(img_2d).save('blur3.png')
