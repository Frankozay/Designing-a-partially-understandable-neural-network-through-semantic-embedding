#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import tensorflow as tf
import random
import numpy as np


def LPF():
    # LPF1 = tf.constant(value=[[0.111111, 0.111111, 0.111111],
    #                           [0.111111, 0.111112, 0.111111],
    #                           [0.111111, 0.111111, 0.111111]],
    #                    dtype=tf.float32)
    LPF1 = tf.constant(value=[[0.1, 0.1, 0.1],
                              [0.1, 0.2, 0.1],
                              [0.1, 0.1, 0.1]],
                       dtype=tf.float32)
    LPF2 = tf.constant(value=[[0.0625, 0.125, 0.0625],
                              [0.0125, 0.25, 0.125],
                              [0.0625, 0.125, 0.0625]],
                       dtype=tf.float32)
    LPF1 = tf.expand_dims(LPF1, 2)
    LPF2 = tf.expand_dims(LPF2, 2)
    LPF1 = tf.expand_dims(LPF1, 3)
    LPF2 = tf.expand_dims(LPF2, 3)
    LPF_ALL = tf.concat([LPF1, LPF2], axis=3)
    return LPF_ALL


def HPF():
    HPF1 = tf.constant(value=[[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]],
                       dtype=tf.float32)
    HPF2 = tf.constant(value=[[-1, -1, -1],
                              [-1, 9, -1],
                              [-1, -1, -1]],
                       dtype=tf.float32)
    HPF3 = tf.constant(value=[[1, -2, 1],
                              [-2, 5, -2],
                              [1, -2, 1]],
                       dtype=tf.float32)
    HPF4 = tf.constant(value=[[1, 1, 1],
                              [1, -7, 1],
                              [1, 1, 1]],
                       dtype=tf.float32)
    HPF1 = tf.expand_dims(HPF1, 2)
    HPF2 = tf.expand_dims(HPF2, 2)
    HPF3 = tf.expand_dims(HPF3, 2)
    HPF4 = tf.expand_dims(HPF4, 2)
    HPF1 = tf.expand_dims(HPF1, 3)
    HPF2 = tf.expand_dims(HPF2, 3)
    HPF3 = tf.expand_dims(HPF3, 3)
    HPF4 = tf.expand_dims(HPF4, 3)
    HPF_ALL = tf.concat([HPF1, HPF2, HPF3, HPF4], axis=3)
    return HPF_ALL


def shift_and_edge_detection():
    SED1 = tf.constant(value=[[0, 0, 0],
                              [-1, 1, 0],
                              [0, 0, 0]],
                       dtype=tf.float32)
    SED2 = tf.constant(value=[[0, -1, 0],
                              [0, 1, 0],
                              [0, 0, 0]],
                       dtype=tf.float32)
    SED3 = tf.constant(value=[[-1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]],
                       dtype=tf.float32)
    SED1 = tf.expand_dims(SED1, 2)
    SED2 = tf.expand_dims(SED2, 2)
    SED3 = tf.expand_dims(SED3, 2)
    SED1 = tf.expand_dims(SED1, 3)
    SED2 = tf.expand_dims(SED2, 3)
    SED3 = tf.expand_dims(SED3, 3)
    SED_ALL = tf.concat([SED1, SED2, SED3], axis=3)
    return SED_ALL


def edge_detection():
    ED1 = tf.constant(value=[[0, -1, 0],
                             [-1, 4, -1],
                             [0, -1, 0]],
                      dtype=tf.float32)
    ED2 = tf.constant(value=[[-1, -1, -1],
                             [-1, 8, -1],
                             [-1, -1, -1]],
                      dtype=tf.float32)
    ED3 = tf.constant(value=[[-0.125, -0.125, -0.125],
                             [-0.125, 1, -0.125],
                             [-0.125, -0.125, -0.125]],
                      dtype=tf.float32)
    ED4 = tf.constant(value=[[1, -2, 1],
                             [-2, 4, -2],
                             [1, -2, 1]],
                      dtype=tf.float32)
    ED5 = tf.constant(value=[[-0.25, 0, 0.25],
                             [0, 0, 0],
                             [0.25, 0, -0.25]],
                      dtype=tf.float32)
    ED6 = tf.constant(value=[[1, 0, 1],
                             [0, -4, 0],
                             [1, 0, 1]],
                      dtype=tf.float32)
    ED1 = tf.expand_dims(ED1, 2)
    ED2 = tf.expand_dims(ED2, 2)
    ED3 = tf.expand_dims(ED3, 2)
    ED4 = tf.expand_dims(ED4, 2)
    ED5 = tf.expand_dims(ED5, 2)
    ED6 = tf.expand_dims(ED6, 2)
    ED1 = tf.expand_dims(ED1, 3)
    ED2 = tf.expand_dims(ED2, 3)
    ED3 = tf.expand_dims(ED3, 3)
    ED4 = tf.expand_dims(ED4, 3)
    ED5 = tf.expand_dims(ED5, 3)
    ED6 = tf.expand_dims(ED6, 3)
    ED_ALL = tf.concat([ED1, ED2, ED3, ED4, ED5, ED6], axis=3)
    return ED_ALL


def embossing_filter():
    EF1 = tf.constant(value=[[-1, -1, 0],
                             [-1, 0, 1],
                             [0, 1, 1]],
                      dtype=tf.float32)
    EF2 = tf.constant(value=[[2, 0, 0],
                             [0, -1, 0],
                             [0, 0, -1]],
                      dtype=tf.float32)
    EF3 = tf.constant(value=[[-2, -1, 0],
                             [-1, 1, 1],
                             [0, 1, 2]],
                      dtype=tf.float32)
    EF1 = tf.expand_dims(EF1, 2)
    EF2 = tf.expand_dims(EF2, 2)
    EF3 = tf.expand_dims(EF3, 2)
    EF1 = tf.expand_dims(EF1, 3)
    EF2 = tf.expand_dims(EF2, 3)
    EF3 = tf.expand_dims(EF3, 3)
    EF_ALL = tf.concat([EF1, EF2, EF3], axis=3)
    return EF_ALL


def EDD():
    EDD1 = tf.constant([[-1, 0, 1],
                        [-1, 0, 1],
                        [-1, 0, 1]], dtype=tf.float32)
    EDD2 = tf.constant([[-1, -1, -1],
                        [0, 0, 0],
                        [1, 1, 1]], dtype=tf.float32)
    EDD1 = tf.expand_dims(EDD1, 2)
    EDD2 = tf.expand_dims(EDD2, 2)
    EDD1 = tf.expand_dims(EDD1, 3)
    EDD2 = tf.expand_dims(EDD2, 3)
    ED_ALL = tf.concat([EDD1, EDD2], axis=3)
    return ED_ALL


def sobel():
    SB1 = tf.constant([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=tf.float32)
    SB2 = tf.constant([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]], dtype=tf.float32)
    SB1 = tf.expand_dims(SB1, 2)
    SB2 = tf.expand_dims(SB2, 2)
    SB1 = tf.expand_dims(SB1, 3)
    SB2 = tf.expand_dims(SB2, 3)
    SB_ALL = tf.concat([SB1, SB2], axis=3)
    return SB_ALL


def scharr():
    SC1 = tf.constant([[3, 0, -3],
                       [10, 0, -10],
                       [3, 0, -3]], dtype=tf.float32)
    SC2 = tf.constant([[3, 10, 3],
                       [0, 0, 0],
                       [-3, -10, -3]], dtype=tf.float32)
    SC1 = tf.expand_dims(SC1, 2)
    SC2 = tf.expand_dims(SC2, 2)
    SC1 = tf.expand_dims(SC1, 3)
    SC2 = tf.expand_dims(SC2, 3)
    SC_ALL = tf.concat([SC1, SC2], axis=3)
    return SC_ALL


def Line_detection():
    LD1 = tf.constant(value=[[-1, -1, -1],
                             [2, 2, 2],
                             [-1, -1, -1]],
                      dtype=tf.float32)
    LD2 = tf.constant(value=[[-1, 2, -1],
                             [-1, 2, -1],
                             [-1, 2, -1]],
                      dtype=tf.float32)
    LD3 = tf.constant(value=[[-1, -1, 2],
                             [-1, 2, -1],
                             [2, -1, -1]],
                      dtype=tf.float32)
    LD4 = tf.constant(value=[[2, -1, -1],
                             [-1, 2, -1],
                             [-1, -1, 2]],
                      dtype=tf.float32)
    LD1 = tf.expand_dims(LD1, 2)
    LD2 = tf.expand_dims(LD2, 2)
    LD3 = tf.expand_dims(LD3, 2)
    LD4 = tf.expand_dims(LD4, 2)
    LD1 = tf.expand_dims(LD1, 3)
    LD2 = tf.expand_dims(LD2, 3)
    LD3 = tf.expand_dims(LD3, 3)
    LD4 = tf.expand_dims(LD4, 3)
    LD_ALL = tf.concat([LD1, LD2, LD3, LD4], axis=3)
    return LD_ALL


def get_fixed_random_filter(in_channels, filter_num):
    # total 28 filters
    global aft0, ft0
    all_filters = tf.concat([
        LPF(), HPF(), shift_and_edge_detection(), edge_detection(), embossing_filter(), EDD(), sobel(), scharr(), Line_detection()],
        axis=3)
    # print(all_filters.shape)
    for i in range(filter_num):
        for j in range(in_channels):
            if j == 0:
                aft0 = all_filters[:, :, :, (i + j) % 28]
                aft0 = tf.expand_dims(aft0, axis=3)
            aft = all_filters[:, :, :, (i + j) % 28]
            aft = tf.expand_dims(aft, axis=3)
            if j != 0:
                aft0 = tf.concat([aft0, aft], axis=2)
            if i == 0 and j == in_channels-1:
                ft0 = aft0
            elif i != 0 and j == in_channels-1:
                ft0 = tf.concat([ft0, aft0], axis=3)

    # print(ft0.shape)
    return ft0


if __name__ == '__main__':
    get_fixed_random_filter(16, 2)
