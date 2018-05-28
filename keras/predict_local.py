

# 验证模型
from keras.models import load_model

from keras.preprocessing.image import ImageDataGenerator
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

img_width, img_height = 400, 400
# test_data_dir = '/search/odin/xudongmei/working/datasets/bak_crop_per_50/test_split'
test_data_dir = r'C:\Users\xudongmei\Desktop\bak\bak_crop\test_split'
nb_train_samples = 26
nb_validation_samples = 8
nb_test_samples = 8
batch_size = 8


best_model = load_model('1.weights-improvement-12-0.88.h5')  # dropout = 0.5
# best_model = load_model("1.weights-improvement-49-0.62.h5")  # dropout = 0.2

# 预测测试集上的平均准确率
datagen = ImageDataGenerator(rescale=1./255)
test_generator = datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle=False,

        class_mode='categorical')

scores = best_model.evaluate_generator(
        test_generator,
        val_samples=nb_test_samples)
print(scores)

predictions = best_model.predict_generator(test_generator, nb_test_samples)
print(predictions)
print('Submission file successfully generated!')

test_image_list = test_generator.filenames
# print(test_image_list)
print('Begin to write submission file ..')
f_submit = open('predict_local_submit.csv', 'w')
f_submit.write('image,baidu,sogou\n')
for i, image_name in enumerate(test_image_list):
    pred = ['%.6f' % p for p in predictions[i, :]]
    if i % 100 == 0:
        print('{} / {}'.format(i, nb_test_samples))
    f_submit.write('%s,%s\n' % (os.path.basename(image_name), ','.join(pred)))

f_submit.close()

print('Submission file successfully generated!')