from keras.models import load_model
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

"""
说明：
    [0.6857351741584695, 0.7145858343337335]
    Begin to write submission file ..
    0 / 420
    100 / 420
    200 / 420
    300 / 420
    400 / 420
    Submission file successfully generated!
    对应submit.csv文件

"""
img_width, img_height = 400, 400
nb_test_samples = 420
batch_size = 8
nbr_augmentation = 1
FishNames = ['baidu_crop', 'sogou_crop']

# root_path = '/Users/pengpai/Desktop/python/DeepLearning/Kaggle/NCFM'
# root_path = '/search/odin/xudongmei/working/projects/task1/Inception_V3'
root_path = r'D:\Ksoftware\PyCharm_Workspace\working\projects\task0\keras'
weights_path = os.path.join(root_path, '1.weights-improvement-12-0.88.h5')


# test_data_dir = "/search/odin/xudongmei/working/datasets/bak_crop_per_50/test_split/"
test_data_dir = r'C:\Users\xudongmei\Desktop\bak\bak_crop_per_50\test_split'

# test data generator for prediction
test_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.1,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)

print('Loading model and weights from training process ...')
model = load_model(weights_path)

for idx in range(nbr_augmentation):
    print('{}th augmentation for testing ...'.format(idx))
    random_seed = np.random.random_integers(0, 100000)

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle=False,  # Important !!!
        seed=random_seed,
        classes=FishNames,
        class_mode='categorical')

    test_image_list = test_generator.filenames
    print(model.metrics_names)  # ['loss', 'acc']
    print(test_generator.class_indices)
    # print('image_list: {}'.format(test_image_list[:10]))
    print('Begin to predict for testing data ...')
    if idx == 0:
        predictions = model.predict_generator(test_generator, nb_test_samples)

        scores = model.evaluate_generator(
            test_generator,
            val_samples=nb_test_samples)
    else:
        predictions += model.predict_generator(test_generator, nb_test_samples)

        scores += model.evaluate_generator(
            test_generator,
            val_samples=nb_test_samples)

predictions /= nbr_augmentation
# scores /= nbr_augmentation
print(scores)

print('Begin to write submission file ..')
f_submit = open(os.path.join(root_path, 'submit.csv'), 'w')
f_submit.write('image,baidu,sogou\n')
for i, image_name in enumerate(test_image_list):
    pred = ['%.6f' % p for p in predictions[i, :]]
    if i % 100 == 0:
        print('{} / {}'.format(i, nb_test_samples))
    f_submit.write('%s,%s\n' % (os.path.basename(image_name), ','.join(pred)))

f_submit.close()

print('Submission file successfully generated!')
