import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 1. 随机调整图片的色彩，定义两种顺序，每次调用的结果均不一样
def distort_color(image, color_ordering=0):
    if color_ordering == 0:
        # 在[-max_delta, max_delta]的范围随机调整图片的亮度
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        # 在[lower, upper]的范围随机调整图的饱和度
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        # 在[-max_delta, max_delta]的范围随机调整图片的色相。max_delta的取值在[0. 0.5]之间
        image = tf.image.random_hue(image, max_delta=0.2)
        # 在[lower, upper]的范围随机调整图的对比度
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32./255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)

    return tf.clip_by_value(image, 0.0, 1.0)


def preprocess_for_train(imgae, height, width, bbox):
    # 如果没有提供标注框，则认为整个图像就是需要关注的部分
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])

    # 转换图像张量的类型
    if imgae.dtype != tf.float32:
        imgae = tf.image.convert_image_dtype(imgae, dtype=tf.float32)

    # 随机截取图像，减小需要关注的物体大小对图像识别算法的影响
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(tf.shape(imgae),
                                                                      bounding_boxes=bbox, min_object_covered=0.4)

    distorted_image = tf.slice(imgae, bbox_begin, bbox_size)

    # 将随机截取的图像调整为神经网络输入层的大小。大小调整的算法是随机选择的。
    distorted_image = tf.image.resize_images(distorted_image, [height, width], method=np.random.randint(4))

    # 随机左右翻转图像
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # 使用一种随机的顺序调整图像色彩
    distorted_image = distort_color(distorted_image, np.random.randint(2))
    return distorted_image


if __name__ == '__main__':
    image_raw_data = tf.gfile.FastGFile(r"C:\Users\xudongmei\Desktop\100米2500pic.北京\
    train_split\baidu2_crop\baidu2-0-0.jpg", "rb").read()
    with tf.Session() as sess:
        img_data = tf.image.decode_jpeg(image_raw_data)

        boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])

        for i in range(9):
            result = preprocess_for_train(img_data, 299, 299, boxes)
            plt.imshow(result.eval())
            plt.show()
