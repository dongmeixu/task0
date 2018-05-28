import os
import numpy as np
import shutil

np.random.seed(2018)

root_train = '/search/odin/xudongmei/working/datasets/bak_crop/train_split'
root_val = '/search/odin/xudongmei/working/datasets/bak_crop/val_split'
root_test = '/search/odin/xudongmei/working/datasets/bak_crop/test_split'

# 总的数据位置
root_total = '/search/odin/xudongmei/working/datasets/bak_crop'

MapNames = ['baidu_crop', 'sogou_crop']

nbr_train_samples = 0
nbr_val_samples = 0
nbr_test_samples = 0

# Training proportion
test_split_proportion = 0.2
val_split_proportion = 0.2

if not os.path.exists(root_train):
    os.mkdir(root_train)

if not os.path.exists(root_val):
    os.mkdir(root_val)

if not os.path.exists(root_test):
    os.mkdir(root_test)

for map in MapNames:
    if map not in os.listdir(root_train):
        os.mkdir(os.path.join(root_train, map))

    if map not in os.listdir(root_val):
        os.mkdir(os.path.join(root_val, map))

    if map not in os.listdir(root_test):
        os.mkdir(os.path.join(root_test, map))

    total_images = os.listdir(os.path.join(root_total, map))

    nbr_test = int(len(total_images) * test_split_proportion)
    nbr_val = int(len(total_images) * val_split_proportion)
    nbr_train = int(len(total_images) - nbr_test - nbr_val)

    # 打乱顺序
    np.random.shuffle(total_images)

    train_images = total_images[:nbr_train]
    val_images = total_images[nbr_train: nbr_train + nbr_val]
    test_images = total_images[nbr_train + nbr_val:]

    for img in train_images:
        source = os.path.join(root_total, map, img)
        target = os.path.join(root_train, map, img)
        shutil.copy(source, target)
        nbr_train_samples += 1

    for img in val_images:
        source = os.path.join(root_total, map, img)
        target = os.path.join(root_val, map, img)
        shutil.copy(source, target)
        nbr_val_samples += 1

    for img in test_images:
        source = os.path.join(root_total, map, img)
        target = os.path.join(root_test, map, img)
        shutil.copy(source, target)
        nbr_test_samples += 1

print('Finish splitting train and val and test images!')
print('# training samples: {}, # val samples: {},  # test samples: {}'.format(nbr_train_samples,
                                                                              nbr_val_samples, nbr_test_samples))
