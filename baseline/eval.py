# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import numpy as np
import tensorflow as tf
import matplotlib.image as mping
import data_process
import model
import train


# test_dir = r'C:\Users\xudongmei\Desktop\bak\bak_crop\split\test_split'
# test_dir = '/search/odin/xudongmei/working/datasets/bak_crop/test_split/'


def evaluate():
    tf.reset_default_graph()
    batch_size = 1

    with tf.Graph().as_default():
        test_images, test_label = data_process.get_files(train.test_dir, 0, False)  # 获取测试图片的地址

        features = tf.placeholder("float32", shape=[batch_size, train.IMG_H, train.IMG_W, train.IMG_C],
                                  name="features")
        labels = tf.placeholder("float32", [batch_size], name="labels")
        one_hot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=train.N_CLASSES)

        train_step, cross_entropy, logits, keep_prob = model.inference(features, one_hot_labels, batch_size,
                                                                       train.N_CLASSES)
        values, indices = tf.nn.top_k(logits, 1)

        accuracy = model.evaluation(logits, one_hot_labels)

        with tf.Session() as sess:
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(train.logs_train_dir)
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
                temp_dict = {}
                image = mping.imread(test_image)
                # print(image.shape)  # (468, 1316, 3)
                image = tf.image.resize_image_with_crop_or_pad(image, train.IMG_H, train.IMG_W)
                x = tf.image.per_image_standardization(image)
                x = sess.run(x)  # 将tensor转化为数组

                predictions = np.squeeze(
                    sess.run(indices, feed_dict={features: np.expand_dims(x, axis=0), keep_prob: 1}), axis=0)
                temp_dict['image_id'] = test_image
                temp_dict['label_id'] = predictions.tolist()
                result.append(temp_dict)
                print('image %s is %d' % (test_image, predictions[0]))

                # 每次预测的准确率
                accuracies = sess.run(accuracy, feed_dict={features: np.expand_dims(x, axis=0),
                                                           labels: np.expand_dims(test_label[i], axis=0),
                                                           keep_prob: 1.0})
                # 计算总的准确率
                total_acc += np.sum(accuracies)

            print(total_acc / len(test_images))

            with open('submit.json', 'w') as f:
                json.dump(result, f)
                print('write result json, num is %d' % len(result))
            import pandas as pd
            df = pd.DataFrame(result)
            df.to_csv("result.csv")


# evaluate()

# 查看checkpoint 内容
import os
model_dir = r'D:\Ksoftware\PyCharm_Workspace\task0_github\baseline\logs\train'
from tensorflow.python import pywrap_tensorflow

checkpoint_path = os.path.join(model_dir, "model.ckpt-999")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tesnsor_name: ", key)
    print(reader.get_tensor(key))
