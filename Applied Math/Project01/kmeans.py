from scipy.spatial.distance import *
import scipy
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from numpy.random import default_rng
from sklearn.cluster import KMeans
import time


def init_centroid(img_1d, k_cluster, init_centroid_type = 'in_pixels'):
    if init_centroid_type == 'in_pixels':
        return img_1d[np.random.choice(img_1d.shape[0], k_cluster, replace= False)]
    if init_centroid_type == 'random':
        return np.random.choice(256, size = (k_cluster, img_1d.shape[1]), replace=False)
        #return np.random.random_integers(low=0, high=255 + 1, size=(k_cluster, img_1d.shape[1]))


def assign_label(img_1d, centroid):
    #norm-2 between pixel and centroid
    distance = np.sqrt(np.sum((img_1d - centroid[0]) ** 2, axis=1))
    distance = distance.reshape((img_1d.shape[0], 1))
    for i in range(1,centroid.shape[0]):
        temp = np.sqrt(np.sum((img_1d - centroid[i]) ** 2, axis=1))
        temp = temp.reshape((img_1d.shape[0], 1))
        distance = np.concatenate((distance,temp),axis=1)
    #return smallest distance's label centroid for each pixel
    return np.argmin(distance,axis = 1)

def update_centroid(img_1d, label, k_cluster, channel):
    centroid = np.zeros((k_cluster,channel))
    for k in range(k_cluster):
        #slice cluster k from img_1d
        cluster_k = img_1d[label == k, :]
        #if cluster have 0 data point -> pass update centroid
        if len(cluster_k) == 0:
            continue
        centroid[k:] = np.mean(cluster_k, axis = 0)
    return centroid

def converge_check(centroid, new_centroid):
    #if distance of value RGB <= Epsilon between old_centroid va new_centroid, assume program has converged
    E = 2
    return np.allclose(centroid, new_centroid, atol = E, equal_nan = True)

def update_data_point(img_1d,k_clusters,label,centroids):
    new_img = np.zeros((img_1d.shape[0],img_1d.shape[1]))
    for k in range(k_clusters):
        new_img[label == k, :] += centroids[k]
    return new_img

def kmeans(img_1d, k_clusters, max_iter = 1000, init_centroids='in_pixels'):
    #flatten array
    row = img_1d.shape[0]
    column = img_1d.shape[1]
    channel = img_1d.shape[2]
    img_1d = img_1d.reshape(img_1d.shape[0] * img_1d.shape[1], img_1d.shape[2])
    #init centroids
    centroid = [init_centroid(img_1d, k_clusters, init_centroids)]
    label = []
    while True and max_iter:
        #assign label of datapoint
        new_label = assign_label(img_1d, centroid[-1])
        label.append(new_label)
        new_centroid = update_centroid(img_1d,label[-1],k_clusters, channel)
        if converge_check(centroid[-1],new_centroid) :
            break
        max_iter-=1
        centroid.append(new_centroid)
    new_img = update_data_point(img_1d,k_clusters,label[-1],centroid[-1])
    new_img = new_img.reshape(row,column,channel)
    return centroid[-1], new_img

if __name__ == '__main__':

    img_1d = Image.open('img_1.jpg')
    img_1d = np.asarray(img_1d)
    k_clusters = 15
    # =50
    row = img_1d.shape[0]
    column = img_1d.shape[1]
    channel = img_1d.shape[2]
    column = img_1d.shape[1]
    channel = img_1d.shape[2]
    #init_centroids = 'in_pixels' OR init_centroids = 'random'
    new_centroid, new_img = kmeans(img_1d, k_clusters,init_centroids = 'in_pixels')
    print('new centroid = ', new_centroid)
    print('new img = ', new_img)
    plt.imshow(new_img.astype('uint8'))
    plt.show()

    img_1d = img_1d.reshape(img_1d.shape[0] * img_1d.shape[1], img_1d.shape[2])
    test = KMeans(n_clusters = k_clusters, random_state=0).fit(img_1d)
    print('sklearn centroid = ', test.cluster_centers_ )
