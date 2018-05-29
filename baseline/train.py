#!/usr/bin/env python
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# sys.path.append("/search/odin/yangyuran/program/Anaconda3/envs/tensorflow/lib/python3.6/site-packages/")


import datetime
import os

import numpy as np
import math
import tensorflow as tf
import data_process
import model
import time
import matplotlib.image as mping
import pandas as pd

from sklearn.preprocessing import OneHotEncoder

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

N_CLASSES = 2
IMG_W = 256  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 256
IMG_C = 3
RATIO = 0.2  # take 20% of dataset as validation data
BATCH_SIZE = 16
CAPACITY = 100 + 3 * BATCH_SIZE
MAX_STEP = 3000  # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.0001  # with current parameters, it is suggested to use learning rate<0.0001

train_dir = '/search/odin/xudongmei/working/datasets/bak2_crop/train_split/'
test_dir = '/search/odin/xudongmei/working/datasets/bak2_crop/test_split/'
logs_train_dir = './logs/train/batch_size=16/'
logs_val_dir = './logs/val/batch_size=16/'
LOGNAME = 'maps'


def run_training():
    with tf.Graph().as_default():
        train, train_label, val, val_label = data_process.get_files(train_dir, RATIO)

        train_batch, train_label_batch = data_process.get_batch(train,
                                                                train_label,
                                                                IMG_W,
                                                                IMG_H,
                                                                BATCH_SIZE,
                                                                CAPACITY)
        val_batch, val_label_batch = data_process.get_batch(val,
                                                            val_label,
                                                            IMG_W,
                                                            IMG_H,
                                                            BATCH_SIZE,
                                                            CAPACITY)

        # 输入数据的命名空间
        with tf.name_scope("Input"):
            features = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, IMG_C], name="X_features")
            labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name="Y_label")
            keep_prob = tf.placeholder(tf.float32, name='Keep_prob')
            one_hot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=N_CLASSES)

        # print(features.shape)
        # print(labels.shape)
        # print(one_hot_labels.shape)

        train_step, cross_entropy, logits, keep_prob = model.inference(features, one_hot_labels, BATCH_SIZE, N_CLASSES)

        accuracy = model.evaluation(logits, one_hot_labels)

        with tf.Session() as sess:
            saver = tf.train.Saver()

            sess.run(tf.global_variables_initializer())

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            summary_op = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
            val_writer = tf.summary.FileWriter(logs_val_dir)

            logger = data_process.train_log(LOGNAME)

            try:
                train_total_loss = 0.0
                train_total_acc = 0.0
                val_total_loss = 0.0
                val_total_acc = 0.0

                for step in np.arange(MAX_STEP):
                    if coord.should_stop():
                        break
                    start_time = time.time()

                    tra_images, tra_labels = sess.run([train_batch, train_label_batch])
                    _, tra_loss, tra_acc = sess.run([train_step, cross_entropy, accuracy],
                                                    feed_dict={features: tra_images, labels: tra_labels,
                                                               keep_prob: 0.5})
                    train_total_loss += tra_loss
                    train_total_acc += tra_acc

                    val_images, val_labels = sess.run([val_batch, val_label_batch])
                    val_loss, val_acc = sess.run([cross_entropy, accuracy],
                                                 feed_dict={features: val_images, labels: val_labels,
                                                            keep_prob: 0.5})
                    val_total_loss += val_loss
                    val_total_acc += val_acc

                    if step % 50 == 0:
                        duration = time.time() - start_time
                        logger.info("step %d: training accuracy %g, loss is %g (%0.3f sec)" % (
                            step, tra_acc, tra_loss, duration))

                        summary_str = sess.run(summary_op,
                                               feed_dict={features: tra_images, labels: tra_labels, keep_prob: 0.5})
                        train_writer.add_summary(summary_str, step)

                    if step % 100 == 0 or (step + 1) == MAX_STEP:
                        duration = time.time() - start_time
                        logger.info("step %d: validation accuracy %g, loss is %g (%0.3f sec)" % (
                            step, tra_acc, tra_loss, duration))

                        summary_str = sess.run(summary_op,
                                               feed_dict={features: val_images, labels: val_labels, keep_prob: 0.5})
                        val_writer.add_summary(summary_str, step)

                    if step % 1000 == 0 or (step + 1) == MAX_STEP:
                        checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=step)

                print("Average loss in training set:", train_total_loss / MAX_STEP)
                print("Average acc in training set:", train_total_acc / MAX_STEP)
                print("Average loss in val set:", val_total_loss / MAX_STEP)
                print("Average acc in val set:", val_total_acc / MAX_STEP)

            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            finally:
                coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    begin = datetime.datetime.now()
    run_training()
    end = datetime.datetime.now()
    print("Training time is: ", end - begin)
