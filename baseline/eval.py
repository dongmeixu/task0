# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.image as mping
import data_process
import model
import train

logs_train_dir = './logs/train/batch_size=16/'


def evaluate():
    tf.reset_default_graph()
    batch_size = 1

    with tf.Graph().as_default():
        test_images, test_label = data_process.get_files(train.test_dir, 0, False)  # 获取测试图片的地址

        features = tf.placeholder("float32", shape=[batch_size, train.IMG_H, train.IMG_W, train.IMG_C],
                                  name="features")
        labels = tf.placeholder("int32", [batch_size], name="labels")
        keep_prob = tf.placeholder(tf.float32, name='Keep_prob')
        one_hot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=train.N_CLASSES)

        train_step, cross_entropy, logits, keep_prob = model.inference(features, one_hot_labels, batch_size,
                                                                       train.N_CLASSES)
        # values, indices = tf.nn.top_k(logits, 1)

        # obtain the probability of each category
        logits = tf.nn.softmax(logits)

        accuracy = model.evaluation(logits, one_hot_labels)

        with tf.Session() as sess:
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print('Restore the model from checkpoint %s' % ckpt.model_checkpoint_path)
                # Restores from checkpoint
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                raise Exception('no checkpoint find')

            result = []
            total_acc = 0.0
            for i, test_image in enumerate(test_images):
                # temp_dict = {}
                temp_dict = collections.OrderedDict()  # 有序字典
                image = mping.imread(test_image)
                # print(image.shape)  # (468, 1316, 3)
                image = tf.image.resize_image_with_crop_or_pad(image, train.IMG_H, train.IMG_W)
                x = tf.image.per_image_standardization(image)
                x = sess.run(x)  # 将tensor转化为数组

                # predictions = np.squeeze(
                # sess.run(indices, feed_dict={features: np.expand_dims(x, axis=0), keep_prob: 1}), axis=0)
                predictions = sess.run(logits, feed_dict={features: np.expand_dims(x, axis=0), keep_prob: 1})
                # print(predictions[0].tolist())  [[0.999958872795105, 4.114030161872506e-05]]
                pred = ['%.6f' % p for p in predictions[0]]
                temp_dict['image_name'] = test_image.split("/")[-1]
                temp_dict['true_label'] = test_label[i]
                temp_dict['p(baidu)'] = pred[0]
                temp_dict['p(sogou)'] = pred[1]

                result.append(temp_dict)
                # print('image %s is %d' % (test_image, predictions[0]))

                # 每次预测的准确率
                accuracies = sess.run(accuracy, feed_dict={features: np.expand_dims(x, axis=0),
                                                           labels: np.expand_dims(test_label[i], axis=0),
                                                           keep_prob: 1.0})
                # 计算总的准确率
                total_acc += np.sum(accuracies)

            print("Final Testing Accurate: ", total_acc / len(test_images))

            df = pd.DataFrame(result)
            df.to_csv("result_bs16.csv")


evaluate()

# # 查看checkpoint 内容
# import os
#
# model_dir = r'D:\Ksoftware\PyCharm_Workspace\task0_github\baseline\logs\train'
# from tensorflow.python import pywrap_tensorflow
#
# checkpoint_path = os.path.join(model_dir, "model.ckpt-999")
# reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
# var_to_shape_map = reader.get_variable_to_shape_map()
# for key in var_to_shape_map:
#     print("tesnsor_name: ", key)
#     print(reader.get_tensor(key))
