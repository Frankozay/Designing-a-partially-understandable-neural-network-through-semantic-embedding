import tensorflow as tf
import tensorflow.contrib.layers as ly
import read_cifar10 as rc
import DataAugmentation as da
import numpy as np
import filter as ft
import sys
import os

path = './model'
unfixed_path = './unfixed_model'
# from cifar10 import *
fixed_rate = 0.125
# Hyperparameter
growth_k = 24
nb_block = 2  # how many (dense block + Transition Layer) ?
lr = 1e-1  # learning_rate
epsilon = 1e-4  # AdamOptimizer epsilon
dropout_rate = 0.2

# Momentum Optimizer will use
nesterov_momentum = 0.9
weight_decay = 1e-4 * (1/(1-fixed_rate))

# Label & batch_size
# batch_size = 64
#
# iteration = 782
# batch_size * iteration = data_set_number
#
# test_iteration = 10
total_epochs = 400
_BATCH_NORM_DECAY = 0.95
_BATCH_NORM_EPSILON = 1e-5
class_num = 10
image_size = 32
img_channels = 3
decay = 1e-4 * (1/(1-fixed_rate))
num_train = 50000
num_test = 10000
train_batch_size = 128
test_batch_size = 200
log_dir = './log/plot_fixed'
unfixed_log_dir = './log/plot_unfixed'
train_log_dir = './log/plot_train'
kernel_name_num = 0

def fixed_padding(inputs, kernel_size):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
    return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides):
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size)

    conv = tf.layers.conv2d(inputs=inputs,
                            filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding=('SAME' if strides == 1 else 'VALID'),
                            use_bias=False,
                            kernel_initializer=ly.xavier_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            data_format='channels_last'
                            )
    return conv


def fixed_conv_fixed_padding(inputs, filters, kernel_size, strides, is_fixed, fixed_rate):
    global kernel_name_num
    fixed_num = int(fixed_rate*filters)
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size)
    random_filter = tf.get_variable(name='random_kernel' + str(kernel_name_num),
                                    shape=[kernel_size, kernel_size, inputs.shape[3].value, filters-fixed_num],
                                    dtype=tf.float32,
                                    initializer=ly.xavier_initializer(),
                                    trainable=True)
    kernel_name_num += 1
    # fixed_filter = tf.constant(value=0.1, dtype=tf.float32,
    #                            shape=[kernel_size, kernel_size, inputs.shape[3].value, filters / 4])
    used_fixed_filter = ft.get_fixed_random_filter(in_channels=inputs.shape[3].value,
                                                   filter_num=fixed_num)
    unuse_fixed_filter = tf.constant(value=0, dtype=tf.float32,
                                     shape=[kernel_size, kernel_size, inputs.shape[3].value, fixed_num])
    fixed_filter = tf.case([(tf.equal(is_fixed, True), lambda: used_fixed_filter),
                            (tf.equal(is_fixed, False), lambda: unuse_fixed_filter)])
    fin_filter = tf.concat([random_filter, fixed_filter], axis=3)

    conv = tf.nn.conv2d(input=inputs,
                        filter=fin_filter,
                        strides=[1, strides, strides, 1],
                        padding=('SAME' if strides == 1 else 'VALID'))

    return conv


def conv_fixed_layer(input, filter, kernel, is_fixed, stride=1, layer_name="fixed_conv"):
    with tf.name_scope(layer_name):
        network = fixed_conv_fixed_padding(input, filter, kernel, stride, is_fixed, fixed_rate)
        return network


def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = conv2d_fixed_padding(input, filter, kernel, stride)
        return network


def batch_norm_relu(inputs, is_training):
    """Performs a batch normalization followed by a ReLU."""
    inputs = tf.layers.batch_normalization(inputs=inputs,
                                           axis=3,
                                           momentum=_BATCH_NORM_DECAY,
                                           epsilon=_BATCH_NORM_EPSILON,
                                           center=True,
                                           scale=True,
                                           training=is_training)
    inputs = tf.nn.relu(inputs)
    return inputs


def Drop_out(x, rate, training):
    return tf.layers.dropout(inputs=x, rate=rate, training=training)


def Average_pooling(x, pool_size=[2, 2], stride=2, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Max_Pooling(x, pool_size=[3, 3], stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Concatenation(layers):
    return tf.concat(layers, axis=3)


# def Evaluate(sess):
#     test_acc = 0.0
#     test_loss = 0.0
#     test_pre_index = 0
#     add = 1000
#
#     for it in range(test_iteration):
#         test_batch_x = test_x[test_pre_index: test_pre_index + add]
#         test_batch_y = test_y[test_pre_index: test_pre_index + add]
#         test_pre_index = test_pre_index + add
#
#         test_feed_dict = {
#             x: test_batch_x,
#             label: test_batch_y,
#             learning_rate: epoch_learning_rate,
#             training_flag: False
#         }
#
#         loss_, acc_ = sess.run([loss, accuracy], feed_dict=test_feed_dict)
#
#         test_loss += loss_ / 10.0
#         test_acc += acc_ / 10.0
#
#     summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
#                                 tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])
#
#     return test_acc, test_loss, summary


class DenseNet():
    def __init__(self, x, nb_blocks, filters, training, is_fixed):
        self.nb_blocks = nb_blocks
        self.filters = filters
        self.training = training
        self.is_fixed = is_fixed
        self.model = self.Dense_net(x)

    def bottleneck_layer(self, x, scope):
        # print(x)
        with tf.name_scope(scope):
            x = batch_norm_relu(x, self.training)
            x = conv_layer(x, filter=4 * self.filters, kernel=1, layer_name=scope + '_conv1')
            x = Drop_out(x, rate=dropout_rate, training=self.training)

            x = batch_norm_relu(x, self.training)
            output_mid = x
            x = conv_fixed_layer(x, filter=self.filters, is_fixed=self.is_fixed, kernel=3, layer_name=scope + '_conv2')
            featuremaps = x
            x = Drop_out(x, rate=dropout_rate, training=self.training)
            # print("bottleneck",x.shape)

            # print(x)

            return output_mid, x, featuremaps

    def transition_layer(self, x, scope):
        with tf.name_scope(scope):
            x = batch_norm_relu(x, self.training)
            # x = conv_layer(x, filter=self.filters, kernel=[1,1], layer_name=scope+'_conv1')

            # https://github.com/taki0112/Densenet-Tensorflow/issues/10

            in_channel = x.shape[-1]
            # print("channel",in_channel)
            x = conv_layer(x, filter=in_channel // 2, kernel=1, layer_name=scope + '_conv1')
            x = Drop_out(x, rate=dropout_rate, training=self.training)
            x = Average_pooling(x, pool_size=[2, 2], stride=2)

            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)

            output, x, featuremaps = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))
            fm = featuremaps
            if layer_name == 'dense_final':
                self.layer1 = tf.identity(output, 'record_layer1')
                self.layer2 = tf.identity(batch_norm_relu(x, self.training), 'record_layer2')

            layers_concat.append(x)

            for i in range(nb_layers - 1):
                x = Concatenation(layers_concat)
                output, x, featuremaps = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
                if layer_name == 'dense_final':
                    if i == 0:
                        self.layer3 = tf.identity(output, 'record_layer3')
                        self.layer4 = tf.identity(batch_norm_relu(x, self.training), 'record_layer4')
                    if i == 1:
                        self.layer5 = tf.identity(output, 'record_layer5')
                        self.layer6 = tf.identity(batch_norm_relu(x, self.training), 'record_layer6')
                    if i == 2:
                        self.layer7 = tf.identity(output, 'record_layer7')
                        self.layer8 = tf.identity(batch_norm_relu(x, self.training), 'record_layer8')

                    if i == 3:
                        self.layer9 = tf.identity(output, 'record_layer9')
                        self.layer10 = tf.identity(batch_norm_relu(x, self.training), 'record_layer10')
                    if i == 4:
                        self.layer11 = tf.identity(output, 'record_layer11')
                        self.layer12 = tf.identity(batch_norm_relu(x, self.training), 'record_layer12')
                    # if i == 5:
                    #     self.layer13 = tf.identity(output, 'record_layer13')
                    #     self.layer14 = tf.identity(batch_norm_relu(x, self.training), 'record_layer14')
                    # if i == 6:
                    #     self.layer15 = tf.identity(output, 'record_layer15')
                    #     self.layer16 = tf.identity(batch_norm_relu(x, self.training), 'record_layer16')
                    # if i == 7:
                    #     self.layer17 = tf.identity(output, 'record_layer17')
                    #     self.layer18 = tf.identity(batch_norm_relu(x, self.training), 'record_layer18')
                    # if i == 8:
                    #     self.layer19 = tf.identity(output, 'record_layer19')
                    #     self.layer20 = tf.identity(batch_norm_relu(x, self.training), 'record_layer20')
                    # if i == 9:
                    #     self.layer21 = tf.identity(output, 'record_layer21')
                    #     self.layer22 = tf.identity(batch_norm_relu(x, self.training), 'record_layer22')
                    # if i == 10:
                    #     self.layer23 = tf.identity(output, 'record_layer23')
                    #     self.layer24 = tf.identity(batch_norm_relu(x, self.training), 'record_layer24')
                    # if i == 11:
                    #     self.layer25 = tf.identity(output, 'record_layer25')
                    #     self.layer26 = tf.identity(batch_norm_relu(x, self.training), 'record_layer26')
                    # if i == 12:
                    #     self.layer27 = tf.identity(output, 'record_layer27')
                    #     self.layer28 = tf.identity(batch_norm_relu(x, self.training), 'record_layer28')
                    # if i == 13:
                    #     self.layer29 = tf.identity(output, 'record_layer29')
                    #     self.layer30 = tf.identity(batch_norm_relu(x, self.training), 'record_layer30')
                    # if i == 14:
                    #     self.layer31 = tf.identity(output, 'record_layer31')
                    #     self.layer32 = tf.identity(batch_norm_relu(x, self.training), 'record_layer32')

                layers_concat.append(x)
            print('len:', len(layers_concat))

            x = Concatenation(layers_concat)

            return x, fm

    def Dense_net(self, input_x):
        # x = conv_layer(input_x, filter=2 * self.filters, kernel= 7, stride=2, layer_name='conv0')
        x = conv_layer(input_x, filter=2 * self.filters, kernel=3, stride=1, layer_name='conv0')
        print(x.shape)
        # x = Max_Pooling(x, pool_size=[3,3], stride=2)

        # for i in range(self.nb_blocks) :
        #     # 6 -> 12 -> 48
        #     x = self.dense_block(input_x=x, nb_layers=4, layer_name='dense_'+str(i))
        #     print(x.shape)
        #     x = self.transition_layer(x, scope='trans_'+str(i))
        #     print('transition_layer', x.shape)

        x, fm = self.dense_block(input_x=x, nb_layers=6, layer_name='dense_1')
        print(x.shape)
        x = self.transition_layer(x, scope='trans_1')
        print(x.shape)

        x, _ = self.dense_block(input_x=x, nb_layers=6, layer_name='dense_2')
        print(x.shape)
        x = self.transition_layer(x, scope='trans_2')
        print(x.shape)

        # x, _ = self.dense_block(input_x=x, nb_layers=6, layer_name='dense_3')
        # print(x.shape)
        # x = self.transition_layer(x, scope='trans_3')
        # print(x.shape)

        x, _ = self.dense_block(input_x=x, nb_layers=6, layer_name='dense_final')
        print(x.shape)

        # 121Layer
        x = batch_norm_relu(x, self.training)
        global_pool = tf.reduce_mean(x, axis=[1, 2], keepdims=False, name='global_pool')
        final_dense = tf.layers.dense(global_pool, class_num, name='final_dense')

        # x = tf.reshape(x, [-1, 10])
        return self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6, self.layer7, self.layer8, \
               self.layer9, self.layer10, self.layer11, self.layer12, final_dense, fm


# =======================================================================================================================


# image_size = 32, img_channels = 3, class_num = 10 in cifar10
# image_size = 32, img_channels = 3, class_num = 10 in cifar10
x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, img_channels])
label = tf.placeholder(tf.float32, shape=[None, class_num])

training_flag = tf.placeholder(tf.bool)
is_fixed = tf.placeholder(tf.bool)
learning_rate = tf.placeholder(tf.float32, name='learning_rate')

_, _, _, _, _, _, _, _, _, _, _, _, final_dense, fm = DenseNet(
    x=x, nb_blocks=nb_block, filters=growth_k, training=training_flag, is_fixed=is_fixed).model

x_train, y_train = rc.Load_traindata()
x_test, y_test = rc.Load_testdata()
cross_entropy = tf.losses.softmax_cross_entropy(logits=final_dense, onehot_labels=label)
loss = cross_entropy + decay * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
correct_prediction = tf.equal(tf.argmax(final_dense, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
counter_paras = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)

optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
updates_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(updates_ops):
    opt_paras = optimizer.minimize(loss, global_step=counter_paras)
# train = optimizer.minimize(loss)
tf.summary.scalar('test_loss', loss)
tf.summary.scalar('pr', accuracy)

saver = tf.train.Saver(tf.global_variables())
reduce_lr_epoch = [60, 160, 260, 350]
max_acc = 0
max_acc_unfixed = 0
merged_all = tf.summary.merge_all()

with tf.Session() as sess:
    fixed_writer = tf.summary.FileWriter(log_dir, sess.graph)
    unfixed_writer = tf.summary.FileWriter(unfixed_log_dir, sess.graph)
    train_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    for epoch in range(total_epochs):
        print('-' * 20, 'epoch', epoch, '-' * 20)
        train_acc = []
        train_loss = []
        test_acc = []
        test_unfixed_acc = []
        # reduce learning rate
        if epoch in reduce_lr_epoch:
            lr = lr * 0.1
            print('reduce learning rate =', lr, 'now')
        # train one epoch
        for iter in range(num_train // train_batch_size):
            # get and preprocess image
            images, labels = rc.Get_batchdata(x_train, y_train, train_batch_size, iter)
            images = da.data_augment(images)
            images = rc.norm_std_each(images)
            # train_one_batch also can accept your own session
            train_feed_dict = {
                x: images,
                label: labels,
                learning_rate: lr,
                training_flag: True,
                is_fixed: True
            }
            _, lossi, acc, merged_train = sess.run([opt_paras, loss, accuracy, merged_all], feed_dict=train_feed_dict)
            train_writer.add_summary(merged_train, epoch * (num_test // test_batch_size) + iter + 1)
            train_acc.append(acc)
            train_loss.append(lossi)
            sys.stdout.write("\r>> train " + str(iter + 1) + '/' + str(num_train // train_batch_size) + ' loss ' + str(
                lossi) + ' acc ' + str(acc))
        mean_train_loss = np.mean(train_loss)
        mean_train_acc = np.mean(train_acc)
        sys.stdout.write("\n")
        print('>> epoch', epoch, 'train mean loss', mean_train_loss, 'train mean acc', mean_train_acc)

        # validate one epoch
        for iter in range(num_test // test_batch_size):
            # get and preprocess image
            images, labels = rc.Get_testbatch(x_test, y_test, test_batch_size, iter)
            images = rc.norm_std_each(images)

            # validate_one_batch also can accept your own session
            # logit, acc = testnet.validate_one_batch(images, labels, epoch * (num_test // test_batch_size) + iter + 1)
            test_feed_dict = {
                x: images,
                label: labels,
                learning_rate: lr,
                training_flag: False,
                is_fixed: True
            }

            loss_, acc_, merged = sess.run([loss, accuracy, merged_all], feed_dict=test_feed_dict)
            fixed_writer.add_summary(merged, epoch * (num_test // test_batch_size) + iter + 1)
            test_acc.append(acc_)

            sys.stdout.write(
                "\r>> fixed_test " + str(iter + 1) + '/' + str(num_test // test_batch_size) + ' acc ' + str(acc_))
        mean_val_acc = np.mean(test_acc)
        if epoch > 120:
            if mean_val_acc > max_acc:
                saver.save(sess, os.path.join(path, "model_share.ckpt"), global_step=counter_paras)
                print('save weight')
                max_acc = mean_val_acc

        sys.stdout.write("\n")
        print('>> epoch', epoch, ' fixed_test mean acc', mean_val_acc, '<<<<<<<mac_acc', max_acc)

        for iter in range(num_test // test_batch_size):
            # get and preprocess image
            images, labels = rc.Get_testbatch(x_test, y_test, test_batch_size, iter)
            images = rc.norm_std_each(images)

            # validate_one_batch also can accept your own session
            # logit, acc = testnet.validate_one_batch(images, labels, epoch * (num_test // test_batch_size) + iter + 1)
            test_feed_dict = {
                x: images,
                label: labels,
                learning_rate: lr,
                training_flag: False,
                is_fixed: False
            }

            loss_, acc_, merged = sess.run([loss, accuracy, merged_all], feed_dict=test_feed_dict)
            unfixed_writer.add_summary(merged, epoch * (num_test // test_batch_size) + iter + 1)
            test_unfixed_acc.append(acc_)

            sys.stdout.write(
                "\r>> unfixed_test " + str(iter + 1) + '/' + str(num_test // test_batch_size) + ' acc ' + str(acc_))
        mean_val_acc = np.mean(test_unfixed_acc)
        if epoch > 120:
            if mean_val_acc > max_acc_unfixed:
                saver.save(sess, os.path.join(unfixed_path, "model_share.ckpt"), global_step=counter_paras)
                print('save weight')
                max_acc_unfixed = mean_val_acc

        sys.stdout.write("\n")
        print('>> epoch', epoch, ' unfixed_test mean acc', mean_val_acc, '<<<<<<<mac_acc', max_acc_unfixed)
    print(max_acc)
    print(max_acc_unfixed)
# =======================================================================================================================

