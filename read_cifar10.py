#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 22:12:34 2018

@author: ddw
"""

import os
import pickle
import numpy as np
# import matplotlib.pyplot as plt
# from scipy.misc import imresize
import DataAugmentation as da

class_num = 10


def get_data_by_label():
    Traindatas, Trainlabels = Load_traindata()
    train_class = np.zeros([10, 5000, 32, 32, 3], dtype='float32')
    train_label = np.zeros([10, 5000, 10])
    cal = np.zeros([10], dtype=int)
    for i in range(50000):
        index = np.argmax(Trainlabels[i])
        train_class[index][cal[index]] = Traindatas[i]
        train_label[index][cal[index]] = Trainlabels[i]
        cal[np.argmax(Trainlabels[i])] += 1

    return train_class, train_label


def get_batch_data_by_label(data, labels, batch_size, i):
    x = np.zeros([batch_size, 32, 32, 3], dtype='float32')
    y = np.zeros([batch_size, class_num], dtype='float32')
    size = data.shape[0]

    if np.random.random() > 0.:
        for i in range(batch_size):
            index = np.random.randint(0, size)
            x[i, ...] = data[index, ...]
            y[i, ...] = labels[index, ...]
    else:
        #        size1 = int(size / batch_size)
        #        index = np.random.randint(0,size1)
        #        x = data[index*batch_size:(index+1)*batch_size,...]
        #        y = labels[index*batch_size:(index+1)*batch_size,...]
        x, y = Getbatchdata_label(data, labels, batch_size, i)

    return x, y


def one_hot(labels):
    labels = np.array(labels)
    return np.eye(class_num)[labels]


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        fo.close()
    return dict


def preprocess(images_1d, dshape=(32, 32, 3)):
    """ Preprocesses CIFAR10 images

    images_1d: np.ndarray
        Unprocessed images
    labels_1d: np.ndarray
        1d vector of labels
    n_labels: int, 10
        Images are split into 10 classes
    dshape: array, [32, 32, 3]
        Images are 32 by 32 RGB
    """
    # Reshape and rotate 1d vector into image
    #    images = rotate_reshape(images_1d, dshape)
    size = images_1d.shape[0]
    channels = np.split(images_1d, 3, 1)
    reshape_channels = [c.reshape([size, 32, 32, 1]) for c in channels]
    images = np.concatenate(reshape_channels, axis=3)

    return images


def rotate_reshape(images, output_shape):
    """ Rotate and reshape n images"""
    # def r_r(img):
    #    """ Rotate and reshape one image """
    #    img = np.reshape(img, output_shape, order="F")
    #    img = np.rot90(img, k=3)
    # new_images = list(map(r_r, images))
    new_images = []
    for img in images:
        img = np.reshape(img, output_shape, order="F")
        img = np.rot90(img, k=3)
        new_images.append(img)
    return new_images


def rescale(images, new_size):
    """ Rescale image to new size"""
    return list(map(lambda img: imresize(img, new_size), images))


def subtract_mean_rgb(images):
    """ Normalize by subtracting from the mean RGB value of all images"""
    return images - np.round(np.mean(images))


def Load_data(file_name, dtype='float64'):
    dict = unpickle(file_name)
    data = dict[b'data'].astype(dtype=dtype)

    labels = dict[b'labels']
    size = data.shape[0]

    data = preprocess(data)
    data = np.reshape(data, [size, 32, 32, 3])

    labels = one_hot(labels)

    return data, labels


def Load_traindata():
    Trainfilename = 'data_batch_'
    Traindatas = []
    Trainlabels = []
    for i in range(1, 6):
        name = Trainfilename + str(i)
        datas, labels = Load_data(name)
        Traindatas.append(datas)
        Trainlabels.append(labels)

    Traindatas = np.reshape(Traindatas, [50000, 32, 32, 3])
    Trainlabels = np.reshape(Trainlabels, [50000, class_num])
    Traindatas = Traindatas

    return Traindatas, Trainlabels


def get_data_by_label():
    Traindatas, Trainlabels = Load_traindata()
    train_class = np.zeros([10, 5000, 32, 32, 3], dtype='float64')
    train_label = np.zeros([10, 5000, 10])
    cal = np.zeros([10], dtype=int)
    for i in range(50000):
        index = np.argmax(Trainlabels[i])
        train_class[index][cal[index]] = Traindatas[i]
        train_label[index][cal[index]] = Trainlabels[i]
        cal[np.argmax(Trainlabels[i])] += 1

    return train_class, train_label


def get_batch_data_by_label(data, labels, batch_size, i):
    x = np.zeros([batch_size, 32, 32, 3], dtype='float32')
    y = np.zeros([batch_size, class_num], dtype='float32')
    size = data.shape[0]

    if np.random.random() > 0.:
        for i in range(batch_size):
            index = np.random.randint(0, size)
            x[i, ...] = data[index, ...]
            y[i, ...] = labels[index, ...]
    else:
        #        size1 = int(size / batch_size)
        #        index = np.random.randint(0,size1)
        #        x = data[index*batch_size:(index+1)*batch_size,...]
        #        y = labels[index*batch_size:(index+1)*batch_size,...]
        x, y = Getbatchdata_label(data, labels, batch_size, i)

    return x, y


def Getbatchdata_label(x_image, y_label, batch_size, i):
    train_size = x_image.shape[0]
    offset = (i * batch_size) % (train_size - batch_size)
    batch_x_image = x_image[offset:(offset + batch_size), ...]
    batch_y_label = y_label[offset:(offset + batch_size), ...]
    batch_x_image = np.reshape(batch_x_image, [batch_size, 32, 32, 3])
    return batch_x_image, batch_y_label


def Load_testdata():
    Testfilename = 'test_batch'
    datas, labels = Load_data(Testfilename)
    datas = datas

    return datas, labels


def norm_std_each(data):
    # norm std
    size = data.shape[0]
    for i in range(size):
        mean_i = np.mean(data[i, ...])
        std_i = np.std(data[i, ...])
        data[i, ...] = (data[i, ...] - mean_i) / std_i
    # data[i,...] = data[i,...] - mean_i
    return data


def Get_batchdata(data, labels, batch_size, i):
    x = np.zeros([batch_size, 32, 32, 3], dtype='float32')
    y = np.zeros([batch_size, class_num], dtype='float32')
    size = data.shape[0]

    if np.random.random() > 0.:
        for i in range(batch_size):
            index = np.random.randint(0, size)
            x[i, ...] = data[index, ...]
            y[i, ...] = labels[index, ...]
    else:
        #        size1 = int(size / batch_size)
        #        index = np.random.randint(0,size1)
        #        x = data[index*batch_size:(index+1)*batch_size,...]
        #        y = labels[index*batch_size:(index+1)*batch_size,...]
        x, y = Getbatchdata(data, labels, batch_size, i)

    return x, y


def Getbatchdata(x_image, y_label, batch_size, i):
    train_size = y_label.shape[0]
    offset = (i * batch_size) % (train_size - batch_size)
    batch_x_image = x_image[offset:(offset + batch_size), ...]
    batch_y_label = y_label[offset:(offset + batch_size), ...]
    batch_x_image = np.reshape(batch_x_image, [batch_size, 32, 32, 3])
    return batch_x_image, batch_y_label


def Get_testbatch(data, label, batch_size, index):
    batch_x_image = data[index * batch_size:(index + 1) * batch_size, ...]
    batch_y_label = label[index * batch_size:(index + 1) * batch_size, ...]
    return batch_x_image, batch_y_label


if __name__ == "__main__":
    filename = 'test_batch'
    data, labels = Load_data(filename)
    data1, labels = Get_batchdata(data, labels, 100, 0)

    plt.imshow(data1[0])
    plt.show()

    data2 = da.img_crop(data1[0])

    #    noise = np.random.normal(loc=0,scale=0.1,size=[10000,32,32,3])
    #    data += noise
    for _ in range(5):
        data2 = da.img_padding_crop(data1[0])
        plt.imshow(data2)
        plt.show()

#    mean = np.mean(data,axis=(0,1,2))
#    data[...,0] = (data[...,0] - mean[0])
#    data[...,0] = data[...,0]/np.std(data[...,0])
#
#    data[...,1] = (data[...,1] - mean[1])
#    data[...,1] = data[...,1]/np.std(data[...,1])
#
#    data[...,2] = (data[...,2] - mean[2])
#    data[...,2] = data[...,2]/np.std(data[...,2])

#    data1,labels = Get_batchdata(data,labels,100)
#
#    data1 = da.data_augment(data1)
#    data1 = norm_std(data1)
#
#    fig = plt.figure()
#    ax = fig.add_subplot(121)
#    img1 = data1[10,...]
#
#    plt.imshow(img1)
#    ax = fig.add_subplot(122)
#    img2 = data1[14,...]
#    plt.imshow(img2)
#    plt.show()
