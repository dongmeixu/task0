#!/usr/bin/python
# -*- coding: UTF-8 -*-
import math
import random
import shutil

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import matplotlib.image as mping
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import logging


"""
将地图不好的那部分截取掉
"""


def cropped_baidu():
    for i in range(21):
        img = Image.open(r"C:\Users\xudongmei\Desktop\bak\bak\baidu\{}.jpg".format(i))
        # 转化为numpy数组，并打印相关信息
        img_ori = np.array(img)
        print("origin image:")
        print("\tshape: ", img_ori.shape)  # HWC img_ori.shape:  (768, 1366, 3)
        print("\tdtype: ", img_ori.dtype)  # img_ori.dtype:  uint8

        """
        截取说明：
            height(768):上下各截掉150
            width(1366):左侧不截，右侧截掉50
        """
        img_crop = img_ori[150:618, 0:1316, :]
        print("cropped image:")
        print("\tshape: ", img_crop.shape)  # shape:  (468, 1316, 3)
        print("\tdtype: ", img_crop.dtype)  # dtype:  uint8

        # plt.figure("Image_{}".format(i))  # 显示图像窗口名称
        # plt.subplot(121)
        # plt.imshow(img_ori)
        # plt.axis("off")  # 关掉坐标轴 默认是开着的
        #
        # plt.subplot(122)
        # plt.imshow(img_crop)
        # plt.axis("off")  # 关掉坐标轴 默认是开着的
        #
        # plt.show()

        img_save = Image.fromarray(img_crop, 'RGB')
        path = r'C:\Users\xudongmei\Desktop\bak\bak\baidu_crop'
        if not os.path.exists(path):
            os.makedirs(path)
        img_save.save(path + '\\baidu_{}.jpg'.format(i))


def cropped_sogou():
    for i in range(21):
        img = Image.open(r"C:\Users\xudongmei\Desktop\bak\bak\sogou\{}.jpg".format(i))
        # 转化为numpy数组，并打印相关信息
        img_ori = np.array(img)
        print("origin image:")
        print("\tshape: ", img_ori.shape)  # HWC img_ori.shape:  (768, 1366, 3)
        print("\tdtype: ", img_ori.dtype)  # img_ori.dtype:  uint8

        """
        截取说明：
            height(768):上截掉150，下截掉100   =========》518
            width(1366):左侧截掉400，右侧截掉100 =======》866
        """
        img_crop = img_ori[150:668, 400:1266, :]
        print("cropped image:")
        print("\tshape: ", img_crop.shape)  # shape:  (518, 866, 3)
        print("\tdtype: ", img_crop.dtype)  # dtype:  uint8

        # plt.figure("Image_{}".format(i))  # 显示图像窗口名称
        # plt.subplot(121)
        # plt.imshow(img_ori)
        # plt.axis("off")  # 关掉坐标轴 默认是开着的
        #
        # plt.subplot(122)
        # plt.imshow(img_crop)
        # plt.axis("off")  # 关掉坐标轴 默认是开着的
        #
        # plt.show()

        img_save = Image.fromarray(img_crop, 'RGB')
        path = r'C:\Users\xudongmei\Desktop\bak\bak\sogou_crop'
        if not os.path.exists(path):
            os.makedirs(path)
        img_save.save(path + '\sogou_{}.jpg'.format(i))


def cropped_baidu2():
    for i in range(50):
        for j in range(50):
            img = Image.open(r"C:\Users\xudongmei\Desktop\100米2500pic.北京\baidu2\{}-{}.jpg".format(i, j))
            # 转化为numpy数组，并打印相关信息
            img_ori = np.array(img)
            print("origin image:")
            print("\tshape: ", img_ori.shape)  # HWC img_ori.shape:  (768, 1366, 3)
            print("\tdtype: ", img_ori.dtype)  # img_ori.dtype:  uint8

            """
            截取说明：
                height(768):上下各截掉150
                width(1366):左侧不截，右侧截掉50
            """
            img_crop = img_ori[150:618, 0:1316, :]
            print("cropped image:")
            print("\tshape: ", img_crop.shape)  # shape:  (468, 1316, 3)
            print("\tdtype: ", img_crop.dtype)  # dtype:  uint8

            # plt.figure("Image_{}".format(i))  # 显示图像窗口名称
            # plt.subplot(121)
            # plt.imshow(img_ori)
            # plt.axis("off")  # 关掉坐标轴 默认是开着的
            #
            # plt.subplot(122)
            # plt.imshow(img_crop)
            # plt.axis("off")  # 关掉坐标轴 默认是开着的
            #
            # plt.show()
            #
            img_save = Image.fromarray(img_crop, 'RGB')
            path = r'C:\Users\xudongmei\Desktop\100米2500pic.北京\baidu_crop'
            if not os.path.exists(path):
                os.makedirs(path)
            img_save.save(path + '\\baidu2-{}-{}.jpg'.format(i, j))


def cropped_sogou2():
    for i in range(60):
        for j in range(60):
            img = Image.open(r"C:\Users\xudongmei\Desktop\100米2500pic.北京\sogou2\{}-{}.jpg".format(i, j))
            # 转化为numpy数组，并打印相关信息
            img_ori = np.array(img)
            print("origin image:")
            print("\tshape: ", img_ori.shape)  # HWC img_ori.shape:  (561, 980, 3)
            print("\tdtype: ", img_ori.dtype)  # img_ori.dtype:  uint8

            """
            截取说明：
                height(561):上截掉0，下截掉50   =========》511
                width(980):左侧截掉20，右侧截掉80 =======》880
            """
            img_crop = img_ori[0:511, 20:900, :]
            print("cropped image:")
            print("\tshape: ", img_crop.shape)  # shape:  (518, 866, 3)
            print("\tdtype: ", img_crop.dtype)  # dtype:  uint8

            # plt.figure("Image_{}".format(i))  # 显示图像窗口名称
            # plt.subplot(121)
            # plt.imshow(img_ori)
            # plt.axis("off")  # 关掉坐标轴 默认是开着的
            #
            # plt.subplot(122)
            # plt.imshow(img_crop)
            # plt.axis("off")  # 关掉坐标轴 默认是开着的
            #
            # plt.show()

            img_save = Image.fromarray(img_crop, 'RGB')
            path = r'C:\Users\xudongmei\Desktop\100米2500pic.北京\sogou2_crop'
            if not os.path.exists(path):
                os.makedirs(path)
            img_save.save(path + '\sogou2-{}-{}.jpg'.format(i, j))


# def get_files(file_dir, test_ratio, val_ratio):
#     """
#     Args:
#         file_dir: file directory
#     Returns:
#         list of images and labels
#     """
#     baidus = []
#     label_baidus = []
#     sogous = []
#     label_sogous = []
#     for file in os.listdir(file_dir):  # file_dir = datasets
#         # print(file)
#         if file == 'baidu_crop':
#             file_list = os.listdir(os.path.join(file_dir, file))
#             # print(file_list) ['0.jpg', '1.jpg', '10.jpg', '11.jpg', '12.jpg', '13.jpg', '14.jpg', '15.jpg',
#             # '16.jpg', '17.jpg', '18.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg', '7.jpg', '8.jpg', '9.jpg']
#
#             for subfile in file_list:  # baidu对应正确的类：1
#                 # print(os.path.join(file_dir, file, subfile))
#                 baidus.append(os.path.join(file_dir, file, subfile))  # 将图片的路径加入到列表中
#                 label_baidus.append(1)  # 添加对应的标签
#             # print(len(baidus))  # 19
#
#         elif file == 'sogou_crop':
#             file_list = os.listdir(os.path.join(file_dir, file))
#             # print(file_list)
#             for subfile in file_list:  # sogou对应正确的类：0
#                 sogous.append(os.path.join(file_dir, file, subfile))
#                 label_sogous.append(0)
#             # print(len(sogous))  # 19
#
#     print('There are %d baidu\nThere are %d sogou' % (len(baidus), len(sogous)))
#
#     image_list = np.hstack((baidus, sogous))
#     # print("type(image_list): ", type(image_list))  # <class 'numpy.ndarray'>
#     # print("image_list.dtype: ", image_list.dtype)
#     # print("image_list.shape: ", image_list.shape)  # (38,)
#     # print("image_list: ", image_list)
#     label_list = np.hstack((label_baidus, label_sogous))
#
#     temp = np.array([image_list, label_list])
#     temp = temp.transpose()
#     # print(temp.shape)  # (38, 2)
#     # print(temp)
#     np.random.shuffle(temp)
#     np.random.shuffle(temp)
#     # print(temp)
#     all_image_list = temp[:, 0]
#     all_label_list = temp[:, 1]
#     #
#     n_sample = len(all_label_list)
#     n_test = math.ceil(n_sample * test_ratio)  # number of testing samples
#     n_train = n_sample - n_test  # number of trainning samples
#     n_val = math.ceil(n_train * val_ratio)  # number of val samples
#
#     tra_images = all_image_list[0:n_train]
#     tra_labels = all_label_list[0:n_train]
#     tra_labels = [int(float(i)) for i in tra_labels]
#     val_images = all_image_list[n_train:-1]
#     val_labels = all_label_list[n_train:-1]
#     val_labels = [int(float(i)) for i in val_labels]
#     print(np.array(tra_images).shape)  # (30, )
#     print(np.array(tra_labels).shape)  # (30, )
#     print(np.array(val_images).shape)  # (7, )
#     print(np.array(val_labels).shape)  # (7, )
#     return tra_images, tra_labels, val_images, val_labels
#
#
# def get_batch(image, label, image_W, image_H, batch_size, capacity):
#     """
#     Args:
#         image: list type
#         label: list type
#         image_W: image width
#         image_H: image height
#         batch_size: batch size
#         capacity: the maximum elements in queue
#     Returns:
#         image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
#         label_batch: 1D tensor [batch_size], dtype=tf.int32
#     """
#
#     image = tf.cast(image, tf.string)
#     label = tf.cast(label, tf.int32)
#
#     # make an input queue
#     input_queue = tf.train.slice_input_producer([image, label])
#
#     label = input_queue[1]
#     image_contents = tf.read_file(input_queue[0])
#     image = tf.image.decode_jpeg(image_contents, channels=3)
#
#     ######################################
#     # data argumentation should go to here
#     ######################################
#
#     image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
#     # if you want to test the generated batches of images, you might want to comment the following line.
#
#     # 如果想看到正常的图片，请注释掉111行（标准化）和 130行（image_batch = tf.cast(image_batch, tf.float32)）
#     # 训练时，不要注释掉！
#     image = tf.image.per_image_standardization(image)
#
#     image_batch, label_batch = tf.train.batch([image, label],
#                                               batch_size=batch_size,
#                                               num_threads=64,
#                                               capacity=capacity)
#
#     label_batch = tf.reshape(label_batch, [batch_size])
#     image_batch = tf.cast(image_batch, tf.float32)
#     print(image_batch.shape, label_batch.shape)  # (2, 208, 208, 3) (2,)
#     print(image_batch.dtype, label_batch.dtype)  # <dtype: 'float32'> <dtype: 'int32'>
#     return image_batch, label_batch


def split_train_test(root_total, root_train, root_test):
    """
    :param root_total
    :param root_train
    :param root_test
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


def get_files(file_dir, val_ratio=0.2, is_train=True):
    """
    :param is_train:
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

    if is_train:  # 如果是训练集的话进行shuffle以及训练与验证集的划分；
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
    else:

        print('# testing samples: {}'.format(len(label_list)))
        return image_list, label_list


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
    image_batch, label_batch = tf.train.batch([image, label],
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


# if __name__ == '__main__':
#     # cropped_baidu()
#     # cropped_sogou()
#
#     # path = r"C:\Users\xudongmei\Desktop\bak\bak_crop"
#     # process_2(path)
#     # aug_path = r"C:\Users\xudongmei\Desktop\bak\bak_crop_per_50"
#     # creat_dataset(img_h=400, img_w=400, path=path, aug_path=aug_path)
#
#     # file_dir = r'C:\Users\xudongmei\Desktop\bak\bak_crop_per_50'
#     # get_files(file_dir, ratio=1)
#     # cropped_baidu2()
#     # cropped_sogou2()
#
#     # root_train = '/media/files/xdm/working/datasets/train_split'
#     # root_val = '/media/files/xdm/working/datasets/val_split'
#     # root_total = '/media/files/xdm/working/datasets/train'
#     #
#     # root_train = r'C:\Users\xudongmei\Desktop\bak\bak_crop\train_split'
#     # root_test = r'C:\Users\xudongmei\Desktop\bak\bak_crop\test_split'
#     #
#     # root_total = r'C:\Users\xudongmei\Desktop\bak\bak_crop'
#     # split_train_test(root_total, root_train, root_test)
#
#     # BATCH_SIZE = 2
#     # CAPACITY = 100 + 3 * BATCH_SIZE
#     # IMG_W = 256
#     # IMG_H = 256
#     #
#     # train_dir = r'C:\Users\xudongmei\Desktop\bak\bak_crop'
#     # val_ratio = 0.2
#     #
#     # tra_images, tra_labels, val_images, val_labels = get_files(train_dir, val_ratio)
#     # tra_images_batch, tra_labels_batch = get_batch(tra_images, tra_labels, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
#     # val_images_batch, val_labels_batch = get_batch(val_images, val_labels, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
#     #
#     # with tf.Session() as sess:
#     #     sess.run(tf.global_variables_initializer())
#     #     i = 0
#     #     coord = tf.train.Coordinator()
#     #     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#     #
#     #     try:
#     #         while not coord.should_stop() and i < 2:
#     #             tra_img, tra_label = sess.run([tra_images_batch, tra_labels_batch])
#     #             for i in range(BATCH_SIZE):
#     #                 print("label: %d" % tra_label[i])
#     #                 plt.imshow(tra_img[i, :, :, :])
#     #                 plt.show()
#     #             i += 1
#     #     except tf.errors.OutOfRangeError:
#     #         print('done!')
#     #
#     #     finally:
#     #         coord.request_stop()
#     #     coord.join(threads)
#
#     # root_train = r'C:\Users\xudongmei\Desktop\100米2500pic.北京\crop\train_split'
#     # root_test = r'C:\Users\xudongmei\Desktop\100米2500pic.北京\crop\test_split'
#     #
#     # root_total = r'C:\Users\xudongmei\Desktop\100米2500pic.北京\crop'
#     # split_train_test(root_total, root_train, root_test)
#
#     path = r'C:\Users\xudongmei\Desktop\100米2500pic.北京\crop\test_split'
#     get_files(path,0.1,False)
