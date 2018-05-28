import datetime

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

train_data_dir = '/search/odin/xudongmei/working/datasets/new_datasets/train_split'
validation_data_dir = '/search/odin/xudongmei/working/datasets/new_datasets/val_split'
test_data_dir = '/search/odin/xudongmei/working/datasets/new_datasets/test_split'
log_dir = '/search/odin/xudongmei/working/projects/task0/logs'

img_width, img_height, img_channel = 256, 256, 3
nb_train_samples = 3360
nb_validation_samples = 1220
nb_test_samples = 1220
epochs = 25
batch_size = 8


def network():
    input_shape = (img_width, img_height, img_channel)

    # 创建CNN
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    return model


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(
    rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# 开始训练
begin = datetime.datetime.now()
from keras.callbacks import ModelCheckpoint, TensorBoard

filepath = "1.weights-improvement-{epoch:02d}-{val_acc:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
tensorboard = TensorBoard(log_dir='logs', histogram_freq=0)
callbacks_list = [checkpoint, tensorboard]

model = network()

H = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=callbacks_list)

# plot the training loss and accuracy
plt.figure()
N = epochs
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")

plt.title("Training Loss and Accuracy on Maps")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
# 存储图像，注意，必须在show之前savefig，否则存储的图片一片空白
# plt.savefig(model_dir + "VGG16_UCM_{}_{}.png".format(batch_size, nbr_epochs))
# plt.savefig(model_dir + "VGG16_2015_4_classes_{}_{}.png".format(batch_size, nbr_epochs))
plt.savefig("cnn_{}_{}.png".format(batch_size, epochs))
# plt.show()

print('[{}]Finishing training...'.format(str(datetime.datetime.now())))

end = datetime.datetime.now()
print("total time: ", end - begin)
