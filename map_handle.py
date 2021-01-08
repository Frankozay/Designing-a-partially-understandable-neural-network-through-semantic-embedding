#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
import os
import cv2
import filter as ft
import matplotlib.pyplot as plt
import DataAugmentation as da
import read_cifar10 as rc
from DenseNet_work import *
import numpy as np

def build_graph():
    # image_size = 32, img_channels = 3, class_num = 10 in cifar10
    sess = tf.Session()
    x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, img_channels])
    label = tf.placeholder(tf.float32, shape=[None, class_num])

    training_flag = tf.placeholder(tf.bool)
    is_fixed = tf.placeholder(tf.bool)
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, final_dense, fm = DenseNet(
        x=x, nb_blocks=nb_block, filters=growth_k, training=training_flag, is_fixed=is_fixed).model

    cross_entropy = tf.losses.softmax_cross_entropy(logits=final_dense, onehot_labels=label)
    loss = cross_entropy + decay * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    correct_prediction = tf.equal(tf.argmax(final_dense, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    counter_paras = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)
    updates_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(updates_ops):
        opt_paras = optimizer.minimize(loss, global_step=counter_paras)
    # train = optimizer.minimize(loss)
    tf.summary.scalar('test_loss', loss)
    tf.summary.scalar('pr', accuracy)
    return x,label,training_flag,is_fixed,learning_rate,fm,final_dense,opt_paras,sess

def model_restore(saver, sess, save_dir):
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(save_dir))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)


def get_map():
    save_dir = 'model/checkpoint'
    train_class, train_label = rc.get_data_by_label()
    x,label,training_flag,is_fixed,learning_rate,fm,final_dense,opt_paras,sess = build_graph()
    print(fm)
    saver = tf.train.Saver()
    saver = tf.train.Saver(write_version=saver_pb2.SaverDef.V1)
    model_restore(saver, sess, save_dir)
    all_map = np.zeros([10, 32, 32, 24], dtype='float32')
    for i in range(10):
        batch_x, batch_y = rc.get_batch_data_by_label(train_class[i], train_label[i], 1, 0)
        feed_dict = {x: batch_x, label: batch_y, learning_rate: 0.1, training_flag: False, is_fixed: True}
        maps = sess.run([fm], feed_dict=feed_dict)
        maps = maps[0]
        maps = np.reshape(maps, [32, 32, 24])
        print(maps.shape)
        all_map[i] = maps
    return all_map


def save_origin_map():
    save_featuremap_origin_dir = './featuremap/origin/'
    for i in range(10):
        if not os.path.exists(save_featuremap_origin_dir + str(i)):
            os.makedirs(save_featuremap_origin_dir + str(i))
    allmaps = get_map()
    for i in range(10):
        tempmaps = allmaps[i]
        for j in range(24):
            maps = tempmaps[:, :, j]
            maps = np.reshape(maps, [32, 32])
            cv2.imwrite(save_featuremap_origin_dir+str(i)+'/'+str(j)+'.png', maps)


def conv():
    save_featuremap_origin_dir = './featuremap/origin/'
    save_featuremap_conv_dir = './featuremap/conv/'
    for i in range(10):
        if not os.path.exists(save_featuremap_conv_dir + str(i)):
            os.makedirs(save_featuremap_conv_dir + str(i))
        for j in range(17):
            if not os.path.exists(save_featuremap_conv_dir + str(i) + '/' + str(j)):
                os.makedirs(save_featuremap_conv_dir + str(i) + '/' + str(j))

    for i in range(10):
        save_featuremap_origin_part_dir = save_featuremap_origin_dir+str(i)+'/'
        for j in range(24):
            img = plt.imread(save_featuremap_origin_part_dir+str(j)+'.png')
            for k in range(17):
                fil = ft.get_filters(k)
                res = cv2.filter2D(img, -1, fil)
                cv2.imwrite(save_featuremap_conv_dir+str(i)+'/'+str(k)+'/'+str(j)+'.png', res*255)

# def show():
#     save_featuremap_conv_dir = './featuremap/conv/0/0/'
#     for i in range(16):
#         img = plt.imread(save_featuremap_conv_dir+str(i)+'.png')
#         cv2.imshow("img", img)
#         cv2.waitKey()
#
#
#
# def test():
#     save_featuremap_origin_dir = './featuremap/origin/'
#     img = plt.imread(save_featuremap_origin_dir+'0/13.png')
#     fil = ft.get_filters(3)
#     res=cv2.filter2D(img,-1,fil)
#     print(res)
#     cv2.imwrite('./test.png',res)


if __name__ == "__main__":
    save_origin_map()
    conv()






