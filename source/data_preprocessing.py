import cv2
import os
import numpy as np
from keras.utils import np_utils

# load du lieu
data_path = r'C:\Users\gsc10\PycharmProjects\Covid19-Prediction\dataset'
categories = os.listdir(data_path)
labels = [i for i in range(len(categories))]

label_dict = dict(zip(categories,labels))
print(label_dict)
data = []
target = []
img_size = 224

# doc du lieu bang opencv va gan nhan cho du lieu
for category in categories:
    folder_path = os.path.join(data_path, category)
    img_names = os.listdir(folder_path)

    for img_name in img_names:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)

        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (img_size, img_size))
            data.append(img)
            target.append(label_dict[category])

        except Exception as e:
            print(e)


data = np.array(data)
target = np.array(target)

# reshape data
data = np.reshape(data, (data.shape[0], img_size, img_size, 1))
target = np_utils.to_categorical(target, 3)

# luu du lieu
np.save("data", data)
np.save("target", target)
