# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import train

'''
Description:
Network structure of a simple CNN network like Alexnet
'''


def weight_variable(shape, stddev=0.1):
    weights = tf.get_variable("weights", shape=shape, dtype=tf.float32,
                              initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
    return weights


def bias_variable(shape, bais=0.1):
    biases = tf.get_variable('biases',
                             shape=shape,
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(bais))
    return biases


def conv2d(x, w):
    return tf.nn.conv2d(x, w, [1, 1, 1, 1], 'SAME', name='conv2')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='max_pool2')


def max_pool_3x3(x):
    return tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME', name='max_pool3')


def avg_pool_3x3(x):
    return tf.nn.avg_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME', name='avg_pool3')


# 修改网络结构-添加可视化卷积层的方法
def inference(features, one_hot_labels, batch_size, n_classes):
    # network structure
    # conv1
    with tf.variable_scope("conv1"):
        W_conv1 = weight_variable([3, 3, 3, 16], stddev=0.1)
        b_conv1 = bias_variable([16])
        h_conv1 = tf.nn.relu(conv2d(features, W_conv1) + b_conv1, name='relu')
        print("after_relu_conv1: ", h_conv1.shape)  # after_relu_conv1:  (32, 256, 256, 16)
        with tf.variable_scope("visualization"):
            x_min = tf.reduce_min(W_conv1)
            x_max = tf.reduce_max(W_conv1)
            kernel_0_to_1 = (W_conv1 - x_min) / (x_max - x_min)
            # to tf.image_summary format [batch_size, height, width, channels]
            kernel_transposed = tf.transpose(kernel_0_to_1, [3, 0, 1, 2])
            # this will display random 3 filters from the 64 in conv1
            tf.summary.image("conv1/filters", kernel_transposed[:, :, :, :3], max_outputs=3)
            layer_image = h_conv1[0: 1, :, :, 0: 16]
            layer_image = tf.transpose(layer_image, perm=[3, 1, 2, 0])
            tf.summary.image("filtered_image_layer", layer_image, max_outputs=16)

    with tf.variable_scope("pooling1_lrn"):
        h_pool1 = max_pool_3x3(h_conv1)
        # norm1
        norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm')
        print("after_pool1: ", h_pool1.shape)  # after_pool1:  (32, 128, 128, 16)

    # conv2
    with tf.variable_scope("conv2"):
        W_conv2 = weight_variable([3, 3, 16, 16], stddev=0.1)
        b_conv2 = bias_variable([16])
        h_conv2 = tf.nn.relu(conv2d(norm1, W_conv2) + b_conv2, name='relu')
        print("after_relu_conv2: ", h_conv2.shape)  # after_relu_conv2:  (32, 128, 128, 16)
        with tf.variable_scope("visualization"):
            x_min = tf.reduce_min(W_conv2)
            x_max = tf.reduce_max(W_conv2)
            kernel_0_to_1 = (W_conv2 - x_min) / (x_max - x_min)
            # to tf.image_summary format [batch_size, height, width, channels]
            kernel_transposed = tf.transpose(kernel_0_to_1, [3, 0, 1, 2])
            # this will display random 3 filters from the 64 in conv1
            tf.summary.image("conv2/filters", kernel_transposed[:, :, :, :3], max_outputs=3)
            layer_image = h_conv2[0: 1, :, :, 0: 16]
            layer_image = tf.transpose(layer_image, perm=[3, 1, 2, 0])
            tf.summary.image("filtered_image_layer", layer_image, max_outputs=16)

    # norm2
    with tf.variable_scope("pooling2_lrn"):
        norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm')
        h_pool2 = max_pool_3x3(norm2)
        print("after_pool2: ", h_pool2.shape)  # after_pool2:  (32, 64, 64, 16)

    # conv3
    with tf.variable_scope("conv3"):
        W_conv3 = weight_variable([3, 3, 16, 64], stddev=0.1)
        b_conv3 = bias_variable([64])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3, name='relu')
        h_pool3 = max_pool_3x3(h_conv3)
        print("after_pool3: ", h_pool3.shape)  # after_pool3:  (32, 32, 32, 64)
        with tf.variable_scope("visualization"):
            x_min = tf.reduce_min(W_conv3)
            x_max = tf.reduce_max(W_conv3)
            kernel_0_to_1 = (W_conv3 - x_min) / (x_max - x_min)
            # to tf.image_summary format [batch_size, height, width, channels]
            kernel_transposed = tf.transpose(kernel_0_to_1, [3, 0, 1, 2])
            # this will display random 3 filters from the 64 in conv1
            tf.summary.image("conv2/filters", kernel_transposed[:, :, :, :3], max_outputs=3)
            layer_image = h_conv3[0: 1, :, :, 0: 16]
            layer_image = tf.transpose(layer_image, perm=[3, 1, 2, 0])
            tf.summary.image("filtered_image_layer", layer_image, max_outputs=16)

    # fc1
    with tf.variable_scope("fc1"):
        reshape = tf.reshape(h_pool3, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        W_fc1 = weight_variable(shape=[dim, 128])
        b_fc1 = bias_variable([128])
        h_fc1 = tf.nn.relu(tf.matmul(reshape, W_fc1) + b_fc1, name='relu')
        print("after_relu_fc1: ", h_fc1.shape)  # after_relu_fc1:  (32, 128)

        # introduce dropout
        keep_prob = tf.placeholder("float")
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='dropout')

    # fc2
    with tf.variable_scope("fc2"):
        W_fc2 = weight_variable([128, n_classes])
        b_fc2 = bias_variable([n_classes])
        y_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        print("after_fc2: ", y_fc2.shape)  # after_fc2:  (32, 2)

    # calculate loss
    with tf.name_scope("Training"):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=y_fc2))
        train_step = tf.train.AdamOptimizer(train.learning_rate).minimize(cross_entropy)
        tf.summary.scalar('cross_entropy', cross_entropy)

    return train_step, cross_entropy, y_fc2, keep_prob


def evaluation(logits, one_hot_labels):
    with tf.name_scope("Evaluation"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        tf.summary.scalar('accuracy', accuracy)
    return accuracy
