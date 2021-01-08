#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 14:04:19 2018

@author: ddw
"""

# In[21]:

from __future__ import print_function

import os

os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
os.environ['TF_CPP_VMODULE'] = "auto_mixed_precision=2"

import numpy as np
from tensorflow.core.protobuf import saver_pb2

import DataAugmentation as da
import read_cifar10 as rc
from Resnet import *

# In[23]:

batch_size = 128
num_enpoch = 1000
size = 20
decay = 1e-4


# In[24]:
def build_graph():
    sess = tf.Session()
    # input
    x_image = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
    y_label = tf.placeholder(dtype=tf.float32, shape=[None, 10])
    drop_keepProb = tf.placeholder(dtype=tf.float32)
    is_training = tf.placeholder(dtype=tf.bool)
    is_fixed = tf.placeholder(dtype=tf.bool)

    # model
    network = Resmodel(20, 10)
    with tf.variable_scope('cnn_paras'):
        y_predict, fin_filters, featuremap = network(x_image, drop_keepProb, is_training, is_fixed)

    print('model builded')
    # final loss
    cross_entropy = tf.losses.softmax_cross_entropy(logits=y_predict, onehot_labels=y_label)
    loss = cross_entropy + decay * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    #    regu_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    #    loss = tf.add_n([cross_entropy]+regu_loss,name="total_loss")
    # test
    correct_prediction = tf.equal(tf.arg_max(input=y_label, dimension=1), tf.arg_max(y_predict, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
    # opt paras
    counter_paras = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
    LR = tf.Variable(0.1, trainable=False, dtype=tf.float32, name='learning_r')
    paras = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='cnn_paras')

    optimizer = tf.train.MomentumOptimizer(learning_rate=LR, momentum=0.9)
    # batch norm
    updates_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(updates_ops):
        opt_paras = optimizer.minimize(loss, global_step=counter_paras)

    tf.summary.scalar('test_loss', loss)
    tf.summary.scalar('pr', accuracy)

    return sess, LR, is_training, x_image, y_label, drop_keepProb, y_predict, loss, accuracy, opt_paras, fin_filters, is_fixed, featuremap


# In[21]:
def Training():
    #    ckpt_dir = './fb_model/'
    #    log_dir = './fb_log/'
    #    save_dir = 'fb_model/checkpoint'

    global merged, merged_unfixed
    save_dir = 'model/checkpoint'
    save_unfixed_dir = './unfixed_model/checkpoint'
    ckpt_dir = './model'
    ckpt_unfixed_dir = './unfixed_model'
    log_dir = './log/'

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.exists(ckpt_unfixed_dir):
        os.makedirs(ckpt_unfixed_dir)

    # read data
    train_data, train_labels = rc.Load_traindata()
    test_data, test_labels = rc.Load_testdata()
    #    train_data = rc.norm_std_each(train_data)
    test_data = rc.norm_std_each(test_data)

    print('read data')

    train_size = train_data.shape[0]
    max_iteration = int(num_enpoch * (train_size // batch_size))

    # graph
    sess, LR, is_training, x_image, y_label, drop_keepProb, y_predict, loss, accuracy, opt_paras, fin_filters, is_fixed, featuremap = build_graph()

    # save model
    saver = tf.train.Saver()
    saver = tf.train.Saver(write_version=saver_pb2.SaverDef.V1)

    merged_all = tf.summary.merge_all()
    # summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
    writer_fixed = tf.summary.FileWriter("./log/plot_fixed", sess.graph)
    writer_unfixed = tf.summary.FileWriter("./log/plot_unfixed", sess.graph)

    sess.run(tf.global_variables_initializer())
    model_restore(saver, sess, save_dir)

    # Training
    accuracy_pre = 0
    accuracy_pre_unfixed = 0
    test_len = 100
    for iter_i in range(65000):
        # tes:
        if iter_i % test_len == 0 and iter_i >= 3000:
            accuracy_temp = 0
            for i in range(5):
                batch_x, batch_y = rc.Get_testbatch(test_data, test_labels, 2000, i)
                feed_dict_test = {x_image: batch_x, y_label: batch_y, drop_keepProb: 1.0, is_training: False,
                                  is_fixed: True}
                accuracy_predict, merged = sess.run([accuracy, merged_all], feed_dict=feed_dict_test)
                # if i == 0:
                #     filters = sess.run([fin_filters], feed_dict=feed_dict_test)
                #     print(filters)
                accuracy_temp += accuracy_predict
            #                print('batch accuracy: %g'% accuracy_predict)

            accuracy_predict = accuracy_temp / 5.0
            writer_fixed.add_summary(merged, iter_i)
            print('fixed test C accuracy: %g,Max accuracy: %g' % (accuracy_predict, accuracy_pre))

            if accuracy_predict > accuracy_pre:
                saver.save(sess, os.path.join(ckpt_dir, "model_share.ckpt"), global_step=iter_i)
                accuracy_pre = accuracy_predict
                print('max accuracy: %g' % accuracy_pre)

            accuracy_temp = 0
            for i in range(5):
                batch_x, batch_y = rc.Get_testbatch(test_data, test_labels, 2000, i)
                feed_dict_test = {x_image: batch_x, y_label: batch_y, drop_keepProb: 1.0, is_training: False,
                                  is_fixed: False}
                accuracy_predict, merged_unfixed = sess.run([accuracy, merged_all], feed_dict=feed_dict_test)
                # if i == 0:
                #     filters = sess.run([fin_filters], feed_dict=feed_dict_test)
                #     print(filters)
                accuracy_temp += accuracy_predict
            #                print('batch accuracy: %g'% accuracy_predict)

            accuracy_predict = accuracy_temp / 5.0
            writer_unfixed.add_summary(merged_unfixed, iter_i)
            print('unfixed test C accuracy: %g,Max accuracy: %g' % (accuracy_predict, accuracy_pre_unfixed))

            if accuracy_predict > accuracy_pre_unfixed:
                saver.save(sess, os.path.join(ckpt_unfixed_dir, "model_share.ckpt"), global_step=iter_i)
                accuracy_pre_unfixed = accuracy_predict
                print('unfixed max accuracy: %g' % accuracy_pre_unfixed)

        # data agrumentation
        batch_x, batch_y = rc.Get_batchdata(train_data, train_labels, batch_size, iter_i)
        batch_x = da.data_augment(batch_x)
        batch_x = rc.norm_std_each(batch_x)

        feed_dict = {x_image: batch_x, y_label: batch_y, drop_keepProb: 1.0, is_training: True, is_fixed: True}

        if iter_i == 400:
            lr = 0.1
            sess.run(tf.assign(LR, lr))
            print('learning rate changed : 0.1')

        if iter_i == 32000:
            test_len = 50
            lr = 0.01
            sess.run(tf.assign(LR, lr))
            print('learning rate changed : 0.01')
        #
        if iter_i == 48000:
            test_len = 20
            lr = 0.001
            sess.run(tf.assign(LR, lr))
            print('learning rate changed : 0.001')

        _, lossi, train_pr = sess.run([opt_paras, loss, accuracy], feed_dict=feed_dict)

        if iter_i % 5 == 0:
            print('step %d: training loss %g : train_pr %g' % (iter_i, lossi, train_pr))

        # if iter_i % 100 == 0:
        #     print(sess.run(fin_filters))


# end


# In[21]:
def model_restore(saver, sess, save_dir):
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(save_dir))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)


def get_noise_batch(x_image, y_label, batch_size, i, noise):
    train_size = y_label.shape[0]
    offset = (i * batch_size) % (train_size - batch_size + 1)
    batch_x_image = x_image[offset:(offset + batch_size), ...]

    batch_x_image += noise

    batch_y_label = y_label[offset:(offset + batch_size), ...]
    batch_x_image = np.reshape(batch_x_image, [batch_size, 32, 32, 1])

    return batch_x_image, batch_y_label


def test():
    # read data
    test_data, test_labels = rc.Load_testdata()
    test_data = rc.norm_std_each(test_data)

    # fb
    save_dir = 'plainnetfb_model1/checkpoint'
    sess, LR, is_training, x_image, y_label, drop_keepProb, y_predict, loss, accuracy, opt_para = build_graph()
    f = open('pr_fb_gaosi.txt', 'w')

    #
    #    save_dir = 'plainnet_model/checkpoint'
    #    sess,LR,is_training,x_image,y_label,drop_keepProb,y_predict,loss,accuracy,opt_para = build_graph()
    ##    f = open('pr_gaosi.txt','w')
    #    f = open('pr_u.txt','w')
    #     graph

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    model_restore(saver, sess, save_dir)

    feed_dict = {x_image: test_data, y_label: test_labels, drop_keepProb: 1.0, is_training: True}
    accuracy_predict = sess.run(accuracy, feed_dict=feed_dict)
    print('test accuracy: {}'.format(accuracy_predict))

    #
    high_value = 0
    noise = np.load('noise.npy')
    #    noise = np.load('noise_u.npy')
    for test_index in range(15):
        #        noise = np.random.uniform(low=0,high=high_value,size=[10000,32,32,3])
        #        noise = np.random.normal(loc=0,scale=high_value,size=[10000,32,32,3])
        test_data = test_data + noise

        feed_dict = {x_image: test_data, y_label: test_labels, drop_keepProb: 1.0, is_training: True}
        accuracy_predict = sess.run(accuracy, feed_dict=feed_dict)

        print('test accuracy{}: {}'.format(test_index, accuracy_predict))

        f.write(str(accuracy_predict))
        f.write('\r\n')

    f.close()


#


# In[21]:
if __name__ == "__main__":
    #    noise = np.random.normal(loc=0,scale=0.05,size=[10000,32,32,3])
    #    noise = np.random.uniform(low=0,high=0.1,size=[10000,32,32,3])
    #    np.save('noise_u',noise)
    #    noise = np.load('noise.npy')
    #    test()
    #    print('done')
    Training()
