#-*- coding:utf-8 -*-
import tensorflow as tf
import os
import cv2
import numpy as np
import re
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()


root_path = 'image/'  # 图片存放的路经
with open('hdf.txt', 'r') as f:  # txt文件中存放的图片路经和相应回归数据
    lines = f.readlines()
num = len(lines)
imgs = np.zeros([num,120000])  # 用于存放图片数据
label = np.zeros([num, 2])
# label_y = np.zeros([num,1]) # 存放标签
for i in range(num):
    line = lines[i]
    segments = re.split('\s+', line)[:-1]
    with tf.gfile.FastGFile(os.path.join(root_path, segments[0]), 'rb') as f1:
        image_data = f1.read()

    image_data = tf.image.decode_jpeg(image_data)
    image_data = tf.reshape(image_data,[1,-1])
    img = sess.run(image_data)

    imgs[i,:]= img
    for j in range(2):
        label[i, j] = float(segments[j + 1])
imgs_mean = np.mean(imgs,axis=0)

# for p in range(num):
#     img2=imgs[p]-imgs_mean
#     imgs[p,:]=img2

train = imgs[:6, :]
train_label = label[:6, :]
test = imgs[6:, :]
test_label = label[6:, :]
img4 = imgs[0]
img4 = img4.reshape([200,200,3])
plt.imshow(img4)
plt.savefig('abc1')
plt.close()
plt.xlim(right=12,left=0)
plt.ylim(top=12,bottom=0)
plt.xticks([])
plt.yticks([])
plt.plot([label[0,0]],[label[0,1]],'ro')
plt.rcParams['figure.figsize'] = (2.0, 2.0)
plt.savefig('abc')
plt.close()