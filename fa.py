#-*- coding:utf-8 -*-
import tensorflow as tf
import os
import cv2
import numpy as np
import re
import matplotlib.pyplot as plt
global_step = 1000
lr = 0.00001
hxw= 67800
sess = tf.InteractiveSession()



def weight_variable(shape):
    # 随机产生一个形状为shape的服从截断正态分布（均值为mean，标准差为stddev）的tensor。
    # 截断的方法根据官方API的定义为:
    # 如果单次随机生成的值偏离均值2倍标准差之外，就丢弃并重新随机生成一个新的数。
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
def conv2d(x, W):
    # strides是指滑动窗口（卷积核）的滑动规则，包含4个维度，分别对应input的4个维度，即每次在input
    # tensor上滑动时的步长。其中batch和in_channels维度一般都设置为1，所以形状为[1, stride, stride, 1]。
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

root_path = 'image/'  # 图片存放的路经
with open('hdf.txt', 'r') as f:  # txt文件中存放的图片路经和相应回归数据
    lines = f.readlines()
num = len(lines)
imgs = np.zeros([num,hxw])  # 用于存放图片数据
label = np.zeros([num, 2])
# label_y = np.zeros([num,1]) # 存放标签
for i in range(num):
    line = lines[i]
    segments = re.split('\s+', line)[:-1]
    img = cv2.imread(os.path.join(root_path, segments[0]))
    img = img.reshape([1, -1])
    # img = cv2.imread(os.path.join(root_path,segments[0],'rb'))
    # img = cv2.transpose()
    imgs[i,:]= img
    for j in range(2):
        label[i, j] = float(segments[j + 1])
imgs_mean = np.mean(imgs,axis=0)
#label_mean = np.mean(label,axis=0)

for p in range(num):
    img2=(imgs[p]-imgs_mean)/255
    imgs[p,:]=img2
# for j in range(num):
#     label2 = label[j]-label_mean
#     label[j,:] = label2
train = imgs[:10, :]
train_label = label[:10, :]
test = imgs[6:, :]
test_label = label[6:, :]
# tu = train[0]
# tu = tu.reshape([200,200,3])
# plt.imshow(tu)
# plt.show()

num1 = len(train)

x = tf.placeholder(tf.float32,shape=[None, hxw])
x_image = tf.reshape(x, [-1, 113, 200, 3])
tf.summary.image('input',x_image,10)
y_ = tf.placeholder(tf.float32, shape=[None, 2])
w_conv1 = weight_variable([5,5,3,64])
b_conv1 = bias_variable([64])
h_conv1 = conv2d(x_image, w_conv1) + b_conv1
w_conv2 = weight_variable([5, 5, 64, 64])
b_conv2 = bias_variable([64])
h_conv2 = conv2d(h_conv1, w_conv2) + b_conv2
w_conv3 = weight_variable([5, 5, 64, 1])
b_conv3 = bias_variable([1])
h_conv3 = conv2d(h_conv2, w_conv3) + b_conv3

w_fc1 = weight_variable([101*188*1, 1024])
b_fc1 = bias_variable([1024])
h_conv3_flat = tf.reshape(h_conv3, [-1, 101*188*1])
h_fc1 = tf.matmul(h_conv3_flat, w_fc1) + b_fc1
w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
h_fc2 = tf.matmul(h_fc1, w_fc2) + b_fc2
w_fc3 = weight_variable([10, 2])
b_fc3 = bias_variable([2])

y = tf.matmul(h_fc2, w_fc3) + b_fc3

loss = tf.reduce_sum(tf.square(y_-y))/num1
train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)


Saver = tf.train.Saver()
try:
    Saver.restore(sess, tf.train.latest_checkpoint("/home/ubuntu/PycharmProjects/yinchengxin/network_model"))
    print('success add the model')
except:
    sess.run(tf.global_variables_initializer())

for step in range(global_step):
  train_step.run(feed_dict={x: train, y_: train_label})


  i_c = str(step)+' '+str(lr)

  accuracy = tf.reduce_mean(abs(y - y_), 0)
  print(i_c+'\t')
  print(accuracy.eval(feed_dict={x: train, y_: train_label}))
  # if step>200 and step%200==0 and step<600:
  #     lr= lr*0.1
Saver.save(sess, "/home/ubuntu/PycharmProjects/yinchengxin/network_model/crack_capcha.model")
# summaries合并
#merged = tf.summary.merge_all()
# 写到指定的磁盘路径中
#train_writer = tf.summary.FileWriter('/train', sess.graph)
#test_writer = tf.summary.FileWriter(log_dir + '/test')

# 运行初始化所有变量
#tf.global_variables_initializer().run()

#correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(abs(y-y_),0)
#sess.run(tf.Print(accuracy,[accuracy]))
print(accuracy.eval(feed_dict={x: train, y_: train_label}))

