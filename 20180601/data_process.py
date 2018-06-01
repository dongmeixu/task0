#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import shutil
import numpy as np
import os
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import logging


# sys.path.append("/search/odin/yangyuran/program/Anaconda3/envs/tensorflow/lib/python3.6/site-packages/")
# TODO:每次分割样本都是一样的，需要不一样嘛
def split_train_val_test(root_total, root_train, root_val, root_test, val_ratio, test_ratio):
    """
    将样本分为训练集、验证集、测试集
    :param root_total 原始数据地址
    :param root_train 训练集保存地址
    :param root_val  验证集保存地址
    :param root_test  测试集保存地址
    :param val_ratio  验证集所占比例
    :param test_ratio  测试集所占比例
    """
    np.random.seed(2018)
    if not os.path.exists(root_train):
        os.mkdir(root_train)

    if not os.path.exists(root_val):
        os.mkdir(root_val)

    if not os.path.exists(root_test):
        os.mkdir(root_test)

    print(os.listdir(root_total))
    Names = os.listdir(root_total)  # 每个文件夹代表一类

    nbr_train_samples = 0
    nbr_val_samples = 0
    nbr_test_samples = 0

    for name in Names:
        # 如果该文件夹不存在，则创建
        if name not in os.listdir(root_train):
            os.mkdir(os.path.join(root_train, name))

        if name not in os.listdir(root_val):
            os.mkdir(os.path.join(root_val, name))

        if name not in os.listdir(root_test):
            os.mkdir(os.path.join(root_test, name))

        total_images = os.listdir(os.path.join(root_total, name))

        nbr_val = int(len(total_images) * val_ratio)
        nbr_test = int(len(total_images) * test_ratio)
        nbr_train = int(len(total_images) - nbr_val - nbr_test)

        # 数据打乱顺序
        np.random.shuffle(total_images)

        train_images = total_images[:nbr_train]
        val_images = total_images[nbr_train: nbr_train + nbr_val]
        test_images = total_images[nbr_train + nbr_val:]

        for img in train_images:
            source = os.path.join(root_total, name, img)
            target = os.path.join(root_train, name, img)
            shutil.copy(source, target)
            nbr_train_samples += 1

        for img in val_images:
            source = os.path.join(root_total, name, img)
            target = os.path.join(root_val, name, img)
            shutil.copy(source, target)
            nbr_val_samples += 1

        for img in test_images:
            source = os.path.join(root_total, name, img)
            target = os.path.join(root_test, name, img)
            shutil.copy(source, target)
            nbr_test_samples += 1

    print('Finish splitting train and test images!')
    print('# training samples: {}, # validation samples: {}, # test samples: {}'
          .format(nbr_train_samples, nbr_val_samples, nbr_test_samples))


def get_files(file_dir):
    """
    返回图像路径列表及其标签列表=====》是按顺序存的
    :param file_dir: str
    :returns the list of images and train's labels
    """
    image_list = []
    label_list = []

    classes = os.listdir(file_dir)
    # print(classes)
    for each in classes:
        print("Starting {} images".format(each))
        for file in os.listdir(os.path.join(file_dir, each)):
            image_list.append(os.path.join(file_dir, each, file))
            label_list.append(each)

    # Encode labels with value between 0 and n_classes-1
    le = LabelEncoder()
    le.fit(label_list)
    label_list = le.fit_transform(label_list)

    return image_list, label_list


def split_train_test(root_total, root_train, root_test):
    """
    将样本分为训练集与测试集
    :param root_total 原始数据地址
    :param root_train 训练集保存地址
    :param root_test  测试集保存地址
    """
    np.random.seed(2018)
    if not os.path.exists(root_train):
        os.mkdir(root_train)

    if not os.path.exists(root_test):
        os.mkdir(root_test)

    print(os.listdir(root_total))
    # Names = ['baidu2_crop', 'sogou2_crop']
    Names = os.listdir(root_total)  # 每个文件夹代表一类

    nbr_train_samples = 0
    nbr_test_samples = 0

    # Training proportion
    split_proportion = 0.8

    for name in Names:
        # 如果该文件夹不存在，则创建
        if name not in os.listdir(root_train):
            os.mkdir(os.path.join(root_train, name))

        if name not in os.listdir(root_test):
            os.mkdir(os.path.join(root_test, name))

        total_images = os.listdir(os.path.join(root_total, name))

        nbr_train = int(len(total_images) * split_proportion)

        # 数据打乱顺序
        np.random.shuffle(total_images)

        train_images = total_images[:nbr_train]

        test_images = total_images[nbr_train:]

        for img in train_images:
            source = os.path.join(root_total, name, img)
            target = os.path.join(root_train, name, img)
            shutil.copy(source, target)
            nbr_train_samples += 1

        for img in test_images:
            source = os.path.join(root_total, name, img)
            target = os.path.join(root_test, name, img)
            shutil.copy(source, target)
            nbr_test_samples += 1

    print('Finish splitting train and test images!')
    print('# training samples: {}, # test samples: {}'.format(nbr_train_samples, nbr_test_samples))


def get_files_by_split_train_val(file_dir, val_ratio=0.2):
    """
    :param file_dir: str
    :param val_ratio: float [0, 1]
    :returns (1)the list of train's images and train's labels
             (2)the list of val's images and val's labels
             (3)the list of test's images and test's labels
    """
    image_list = []
    label_list = []

    classes = os.listdir(file_dir)
    # print(classes)
    for each in classes:
        print("Starting {} images".format(each))
        for file in os.listdir(os.path.join(file_dir, each)):
            image_list.append(os.path.join(file_dir, each, file))
            label_list.append(each)

    # Encode labels with value between 0 and n_classes-1
    le = LabelEncoder()
    le.fit(label_list)
    label_list = le.fit_transform(label_list)

    # 对训练集进行shuffle以及训练与验证集的划分
    # 利用shuffle打乱顺序
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    # print(temp.shape)  # (38, 2)

    # 从打乱的temp中再取出list(img和label)
    all_image_list = temp[:, 0]
    all_label_list = temp[:, 1]

    # number of all samples
    n_sample = len(all_label_list)
    # number of val samples
    n_val = math.ceil(n_sample * val_ratio)
    # number of training samples
    n_train = n_sample - n_val

    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]

    val_images = all_image_list[n_train:]
    val_labels = all_label_list[n_train:]
    val_labels = [int(float(i)) for i in val_labels]
    # TODO: 1.将对应关系保存下来，因为存在shuffle操作每次的数据是不一样的@2018/5/23/11:38
    # TODO: 2. 制作成TFrecode的格式呢？
    print('# training samples: {}, # validation samples: {}'.format(len(tra_labels), len(val_labels)))
    return tra_images, tra_labels, val_images, val_labels


def get_batch(image, label, image_W, image_H, batch_size, capacity):
    """
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    """
    """
    step1：
        将上面生成的List传入get_batch() ，
        转换类型，产生一个输入队列queue，因为img和lab是分开的，
        所以使用tf.train.slice_input_producer()，
        然后用tf.read_file()从队列中读取图像
    """

    ########################################################
    # step1：
    #   将上面生成的List传入get_batch() ，
    #   转换类型，产生一个输入队列queue，因为img和lab是分开的，
    #   所以使用tf.train.slice_input_producer()，
    #   然后用tf.read_file()
    #   从队列中读取图像
    ########################################################
    # print(image)
    # seed = np.random.seed(2018)
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    # 输入队列中原始的元素为文件列表中的所有文件
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    # read img from a queue
    image_contents = tf.read_file(input_queue[0])

    ########################################################
    # step2:将图像解码，不同类型的图像不能混在一起
    ########################################################
    image = tf.image.decode_jpeg(image_contents, channels=3)

    ########################################################
    # step3: data argumentation should go to here
    ########################################################
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    # if you want to test the generated batches of images, you might want to comment the following line.

    # 如果想看到正常的图片，请注释掉111行（标准化）和 130行（image_batch = tf.cast(image_batch, tf.float32)）
    # 训练时，不要注释掉！
    image = tf.image.per_image_standardization(image)
    # print(image, label)
    ########################################################
    # step4: 生成batch
    ########################################################
    image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                      batch_size=batch_size,
                                                      num_threads=64,
                                                      capacity=capacity)
    # 重新排列label，行数为[batch_size]
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    # print(image_batch.shape, label_batch.shape)  # (2, 208, 208, 3) (2,)
    # print(image_batch.dtype, label_batch.dtype)  # <dtype: 'float32'> <dtype: 'int32'>
    return image_batch, label_batch


def train_log(filename='logfile'):
    # create logger
    logger_name = "filename"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # create file handler
    log_path = './' + filename + '.log'
    fh = logging.FileHandler(log_path)
    ch = logging.StreamHandler()

    # create formatter
    fmt = "%(asctime)-15s %(levelname)s %(filename)s %(lineno)d %(process)d %(message)s"
    datefmt = "%a %d %b %Y %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt)

    # add handler and formatter to logger
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


if __name__ == '__main__':
    # cropped_baidu()
    # cropped_sogou()

    # path = r"C:\Users\xudongmei\Desktop\bak\bak_crop"
    # process_2(path)
    # aug_path = r"C:\Users\xudongmei\Desktop\bak\bak_crop_per_50"
    # creat_dataset(img_h=400, img_w=400, path=path, aug_path=aug_path)

    # file_dir = r'C:\Users\xudongmei\Desktop\bak\bak_crop_per_50'
    # get_files(file_dir, ratio=1)
    # cropped_baidu2()
    # cropped_sogou2()

    # root_train = '/media/files/xdm/working/datasets/train_split'
    # root_val = '/media/files/xdm/working/datasets/val_split'
    # root_total = '/media/files/xdm/working/datasets/train'
    #
    # root_train = r'C:\Users\xudongmei\Desktop\bak\bak_crop\train_split'
    # root_test = r'C:\Users\xudongmei\Desktop\bak\bak_crop\test_split'
    #
    # root_total = r'C:\Users\xudongmei\Desktop\bak\bak_crop'
    # split_train_test(root_total, root_train, root_test)

    # BATCH_SIZE = 2
    # CAPACITY = 100 + 3 * BATCH_SIZE
    # IMG_W = 256
    # IMG_H = 256
    #
    # train_dir = r'C:\Users\xudongmei\Desktop\bak\bak_crop'
    # val_ratio = 0.2
    #
    # tra_images, tra_labels, val_images, val_labels = get_files(train_dir, val_ratio)
    # tra_images_batch, tra_labels_batch = get_batch(tra_images, tra_labels, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
    # val_images_batch, val_labels_batch = get_batch(val_images, val_labels, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
    #
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     i = 0
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #
    #     try:
    #         while not coord.should_stop() and i < 2:
    #             tra_img, tra_label = sess.run([tra_images_batch, tra_labels_batch])
    #             for i in range(BATCH_SIZE):
    #                 print("label: %d" % tra_label[i])
    #                 plt.imshow(tra_img[i, :, :, :])
    #                 plt.show()
    #             i += 1
    #     except tf.errors.OutOfRangeError:
    #         print('done!')
    #
    #     finally:
    #         coord.request_stop()
    #     coord.join(threads)

    # root_train = r'C:\Users\xudongmei\Desktop\100米2500pic.北京\crop\train_split'
    # root_test = r'C:\Users\xudongmei\Desktop\100米2500pic.北京\crop\test_split'
    #
    # root_total = r'C:\Users\xudongmei\Desktop\100米2500pic.北京\crop'
    # split_train_test(root_total, root_train, root_test)

    # total = r'C:\Users\xudongmei\Desktop\bak\bak_crop\origin'
    # train = r'C:\Users\xudongmei\Desktop\bak\bak_crop\train_split'
    # val = r'C:\Users\xudongmei\Desktop\bak\bak_crop\val_split'
    # test = r'C:\Users\xudongmei\Desktop\bak\bak_crop\test_split'

    total = r'C:\Users\xudongmei\Desktop\100米2500pic.北京\crop'
    train = r'C:\Users\xudongmei\Desktop\100米2500pic.北京\train_split'
    val = r'C:\Users\xudongmei\Desktop\100米2500pic.北京\val_split'
    test = r'C:\Users\xudongmei\Desktop\100米2500pic.北京\test_split'

    ratio = 0.1
    split_train_val_test(total, train, val, test, ratio, ratio)
