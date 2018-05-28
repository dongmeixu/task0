# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import numpy as np
import tensorflow as tf
import data_process
import model
import train


test_dir = 'datasets/bak_crop/'

def evaluate():
    with tf.Graph().as_default():
        test_images, test_label = data_process.get_files(test_dir, 0, False)
        features = tf.placeholder("float32", shape=[train.BATCH_SIZE, train.IMG_H, train.IMG_W, train.IMG_C], name="features")
        labels = tf.placeholder("float32", [train.BATCH_SIZE], name="labels")
        one_hot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
        train_step, cross_entropy, logits, keep_prob = model.inference(features, one_hot_labels, train.BATCH_SIZE, 2)
        values, indices = tf.nn.top_k(logits, 1)

        with tf.Session() as sess:
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(train.logs_train_dir)
            print(ckpt.model_checkpoint_path)
            if ckpt and ckpt.model_checkpoint_path:
                print('Restore the model from checkpoint %s' % ckpt.model_checkpoint_path)
                # Restores from checkpoint
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                raise Exception('no checkpoint find')

            result = []
            for test_image in test_images:
                temp_dict = {}

                image = tf.image.resize_image_with_crop_or_pad(test_image, train.IMG_H, train.IMG_W)
                x = tf.image.per_image_standardization(image)
                predictions = np.squeeze(
                    sess.run(indices, feed_dict={features: np.expand_dims(x, axis=0), keep_prob: 1}), axis=0)
                temp_dict['image_id'] = test_image
                temp_dict['label_id'] = predictions.tolist()
                result.append(temp_dict)
                print('image %s is %d,%d,%d' % (test_image, predictions[0]))

            with open('submit.json', 'w') as f:
                json.dump(result, f)
                print('write result json, num is %d' % len(result))


evaluate()