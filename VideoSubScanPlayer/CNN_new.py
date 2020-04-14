# coding=utf-8
import tensorflow as tf
import os
import numpy as np
import random
from PIL import Image
import cv2 as cv
import time

"""
各类模型代码以及训练代码
"""

image_size = 64
# all:33-127, 0-9:48-58
start = 48
end = 58
category = end - start

# 用于存放数据集的字典，字典格式 路径:ascii值
train_list = []
test_list = []
train_rate = 0.90 #13818
# 提供数据集路径
file_dir = "G:\\Code\\Python\\digital_imgae_processing_design\\dataset\\"
asciiList = [str(index) for index in range(start, end)]
# 获取所有数据的路径存在file_list字典
for index, numStr in enumerate(asciiList):
	file_name_list = os.listdir(file_dir + numStr + '\\')
	file_list = []
	for name in file_name_list:
		file_list.append([name, index + start])
	# 打乱
	random.shuffle(file_list)
	# 按分配率来分配
	train_length = int(len(file_list) * train_rate)
	train_list += file_list[0: train_length + 1]
	test_list += file_list[train_length + 1:]



X = tf.placeholder(tf.float32, [None, 64 * 64])
Y = tf.placeholder(tf.float32, [None, category])
keep_prob = tf.placeholder(tf.float32)


def getRandomData(sample_list,batch_num):
	# 传入batch_num为一次性返回的样本标签数量
	# 随机获取指定数量的image和label
	# 返回的数据  image为one-hot np-array,维度 【batch_num, image总像素】
	#           label为01矩阵,维度[batch_num, classes] 其中正确标签为1，

	image_batch = []
	label_batch = []

	# 从读取好的字典中随机读取指定数量的数据
	elements = random.sample(sample_list, batch_num)
	# print(elements)
	for item in elements:
		name = item[0]
		id = item[1]
		# 将读取的图片转换为nd-array格式，并将长度reshape成一维向量
		# 图片先用Image工具读取，再用numpy转换，然后转换为二值图，再定为one-hot
		img = Image.open(file_dir + str(id) + "\\" + name)
		image_batch.append(Nomalize(np.array(img.convert("L")), 128).reshape([image_size ** 2]))

		label_array = [0] * (end - start)
		label_array[id - start] = 1
		label_batch.append(label_array)

	# 将转换好的元素转换为nd-array格式
	image_batch = np.array(image_batch)
	label_batch = np.array(label_batch)

	# print(image_batch.shape)
	# print(label_batch.shape)

	return image_batch, label_batch

def getDataBatch(sample_list, batch_num):
	# 传入batch_num为一次性返回的样本标签数量
	# 获取指定长度的数量
	# 返回的数据  image为one-hot np-array,维度 【batch_num, image总像素】
	#           label为01矩阵,维度[batch_num, classes] 其中正确标签为1，
	# 注： 不断出队确保了每次取样本都不会重复，但是别越界了。
	# 测试情况下，94 * 21 = 1974样本，总样本不要超过这个值

	image_batch = []
	label_batch = []

	# 从读取好的字典中读取指定数量的数据,按顺序
	elements = []
	for i in range(batch_num):
		elements.append(sample_list.pop(0))

	for item in elements:
		name = item[0]
		id = item[1]
		# 将读取的图片转换为nd-array格式，并将长度reshape成一维向量
		# 图片先用Image工具读取，再用numpy转换，然后转换为二值图，再定为one-hot
		img = Image.open(file_dir + str(id) + "\\" + name)
		image_batch.append(Nomalize(np.array(img.convert("L")), 128).reshape([image_size ** 2]))

		label_array = [0] * (end - start)
		label_array[id - start] = 1
		label_batch.append(label_array)

	# 将转换好的元素转换为nd-array格式
	image_batch = np.array(image_batch)
	label_batch = np.array(label_batch)

	# print(image_batch.shape)
	# print(label_batch.shape)

	return image_batch, label_batch

def getRealData():
	# 获取真实分割的测试数据
	test_dir = "chuli\\"
	file_name_list = os.listdir(test_dir)
	data = []
	for id, name in enumerate(file_name_list):
		image_path = test_dir + name
		img = Image.open(image_path)
		img_np = Nomalize(np.array(img.convert("L")), 128).reshape([image_size ** 2])
		data.append([img_np,id])

	return data


def convert_pic(imgPath):
	# 把图片转为numpy 【64*64】
	img = Image.open(imgPath)
	print(img)
	row,col = img.size
	img = cv.imread(imgPath,0)
	cv.imshow('img',img)
	# image = cv.cvtColor(np.asarray(img), cv.COLOR_RGB2GRAY)
	# rate = int((64*row)/col)
	# image = cv.resize(image,(0,0),fx = row*rate, fy = 64)
	# cv.imshow('img',image)
	cv.waitKey(0)

def Nomalize(array, value):
	row, col = array.shape
	for i in range(row):
		for j in range(col):
			if array[i][j] > value:
				array[i][j] = 1
			else:
				array[i][j] = 0
	return array

def ascii_cnn():
	x = tf.reshape(X, shape = [-1, 64, 64, 1])
	# 2 conv layers
	w_c1 = tf.Variable(tf.random_normal([3, 3, 1, 64], stddev = 0.01))
	b_c1 = tf.Variable(tf.zeros([64]))
	conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides = [1, 1, 1, 1], padding = 'SAME'), b_c1))
	conv1 = tf.nn.max_pool(conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

	w_c2 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev = 0.01))
	b_c2 = tf.Variable(tf.zeros([128]))
	conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides = [1, 1, 1, 1], padding = 'SAME'), b_c2))
	conv2 = tf.nn.max_pool(conv2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

	# 16*16*64
	# w_c3 = tf.Variable(tf.random_normal([3, 3, 128, 256], stddev = 0.01))
	# b_c3 = tf.Variable(tf.zeros([256]))
	# conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides = [1, 1, 1, 1], padding = 'SAME'), b_c3))
	# conv3 = tf.nn.max_pool(conv3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
	# conv3 = tf.nn.dropout(conv3, keep_prob)

	# 全连接层，8*8*128
	w_d = tf.Variable(tf.random_normal([16 * 16 * 128, 1024], stddev = 0.01))
	b_d = tf.Variable(tf.zeros([1024]))
	dense = tf.reshape(conv2, [-1, w_d.get_shape().as_list()[0]])
	dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
	dense = tf.nn.dropout(dense, keep_prob)

	w_out = tf.Variable(tf.random_normal([1024, 94], stddev = 0.01))
	b_out = tf.Variable(tf.zeros([94]))
	out = tf.add(tf.matmul(dense, w_out), b_out)

	return out


def M4_sub():
	x = tf.reshape(X, shape = [-1, 64, 64, 1])
	# 2 conv layers
	w_c1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev = 0.01))
	b_c1 = tf.Variable(tf.zeros([32]))
	conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides = [1, 1, 1, 1], padding = 'SAME'), b_c1))
	conv1 = tf.nn.max_pool(conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

	# 32*32*64
	w_c2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev = 0.01))
	b_c2 = tf.Variable(tf.zeros([64]))
	conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides = [1, 1, 1, 1], padding = 'SAME'), b_c2))
	conv2 = tf.nn.max_pool(conv2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

	# 全连接层，16*16*128
	w_d = tf.Variable(tf.random_normal([16 * 16 * 64, 1024], stddev = 0.01))
	b_d = tf.Variable(tf.zeros([1024]))
	dense = tf.reshape(conv2, [-1, w_d.get_shape().as_list()[0]])
	dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
	dense = tf.nn.dropout(dense, keep_prob)

	w_out = tf.Variable(tf.random_normal([1024, category], stddev = 0.01))
	b_out = tf.Variable(tf.zeros([category]))
	out = tf.add(tf.matmul(dense, w_out), b_out)

	return out

def M4_plus():
	x = tf.reshape(X, shape = [-1, 64, 64, 1])
	# 输入64 * 64 卷积层 C1
	w_c1 = tf.Variable(tf.random_normal([3, 3, 1, 64], stddev = 0.01))
	b_c1 = tf.Variable(tf.zeros([64]))
	conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides = [1, 1, 1, 1], padding = 'SAME'), b_c1))
	conv1 = tf.nn.max_pool(conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

	# 32*32*64 卷积层 C2
	w_c2 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev = 0.01))
	b_c2 = tf.Variable(tf.zeros([128]))
	conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides = [1, 1, 1, 1], padding = 'SAME'), b_c2))
	conv2 = tf.nn.max_pool(conv2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

	# 全连接层，16*16*128
	w_d = tf.Variable(tf.random_normal([16 * 16 * 128, 1024], stddev = 0.01))
	b_d = tf.Variable(tf.zeros([1024]))
	dense = tf.reshape(conv2, [-1, w_d.get_shape().as_list()[0]])
	dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
	dense = tf.nn.dropout(dense, keep_prob)
	# 全连接层，1024 -> 90
	w_out = tf.Variable(tf.random_normal([1024, category], stddev = 0.01))
	b_out = tf.Variable(tf.zeros([category]))
	out = tf.add(tf.matmul(dense, w_out), b_out)

	return out

def M5_plus():
	# M5+ 64-128-256-2fc
	x = tf.reshape(X, shape = [-1, 64, 64, 1])
	# 3 conv layers
	w_c1 = tf.Variable(tf.random_normal([3, 3, 1, 64], stddev = 0.01))
	b_c1 = tf.Variable(tf.zeros([64]))
	conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides = [1, 1, 1, 1], padding = 'SAME'), b_c1))
	conv1 = tf.nn.max_pool(conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

	# 32*32*64
	w_c2 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev = 0.01))
	b_c2 = tf.Variable(tf.zeros([128]))
	conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides = [1, 1, 1, 1], padding = 'SAME'), b_c2))
	conv2 = tf.nn.max_pool(conv2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

	# 16*16*128
	w_c3 = tf.Variable(tf.random_normal([3, 3, 128, 256], stddev = 0.01))
	b_c3 = tf.Variable(tf.zeros([256]))
	conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides = [1, 1, 1, 1], padding = 'SAME'), b_c3))
	conv3 = tf.nn.max_pool(conv3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
	conv3 = tf.nn.dropout(conv3, keep_prob)

	# 全连接层，8*8*256
	w_d = tf.Variable(tf.random_normal([8 * 8 * 256, 1024], stddev = 0.01))
	b_d = tf.Variable(tf.zeros([1024]))
	dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
	dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
	dense = tf.nn.dropout(dense, keep_prob)

	w_out = tf.Variable(tf.random_normal([1024, category], stddev = 0.01))
	b_out = tf.Variable(tf.zeros([category]))
	out = tf.add(tf.matmul(dense, w_out), b_out)

	return out

def M5_sub():
	# M5- 32-64-128-2fc
	x = tf.reshape(X, shape = [-1, 64, 64, 1])
	# 3 conv layers
	w_c1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev = 0.01))
	b_c1 = tf.Variable(tf.zeros([32]))
	conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides = [1, 1, 1, 1], padding = 'SAME'), b_c1))
	conv1 = tf.nn.max_pool(conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

	# 32*32*32
	w_c2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev = 0.01))
	b_c2 = tf.Variable(tf.zeros([64]))
	conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides = [1, 1, 1, 1], padding = 'SAME'), b_c2))
	conv2 = tf.nn.max_pool(conv2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

	# 16*16*64
	w_c3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev = 0.01))
	b_c3 = tf.Variable(tf.zeros([128]))
	conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides = [1, 1, 1, 1], padding = 'SAME'), b_c3))
	conv3 = tf.nn.max_pool(conv3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
	conv3 = tf.nn.dropout(conv3, keep_prob)

	# 全连接层，8*8*128
	w_d = tf.Variable(tf.random_normal([8 * 8 * 128, 1024], stddev = 0.01))
	b_d = tf.Variable(tf.zeros([1024]))
	dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
	dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
	dense = tf.nn.dropout(dense, keep_prob)

	w_out = tf.Variable(tf.random_normal([1024, category], stddev = 0.01))
	b_out = tf.Variable(tf.zeros([category]))
	out = tf.add(tf.matmul(dense, w_out), b_out)

	return out

def M6_sub():
	# M6- 32-64-128-256-2fc
	x = tf.reshape(X, shape = [-1, 64, 64, 1])
	# 4 conv layers
	w_c1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev = 0.01))
	b_c1 = tf.Variable(tf.zeros([32]))
	conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides = [1, 1, 1, 1], padding = 'SAME'), b_c1))
	conv1 = tf.nn.max_pool(conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

	# 32*32*32
	w_c2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev = 0.01))
	b_c2 = tf.Variable(tf.zeros([64]))
	conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides = [1, 1, 1, 1], padding = 'SAME'), b_c2))
	conv2 = tf.nn.max_pool(conv2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

	# 16*16*64
	w_c3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev = 0.01))
	b_c3 = tf.Variable(tf.zeros([128]))
	conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides = [1, 1, 1, 1], padding = 'SAME'), b_c3))
	conv3 = tf.nn.max_pool(conv3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
	conv3 = tf.nn.dropout(conv3, keep_prob)

	# 8*8*128
	w_c4 = tf.Variable(tf.random_normal([3, 3, 128, 256], stddev = 0.01))
	b_c4 = tf.Variable(tf.zeros([256]))
	conv4 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv3, w_c4, strides = [1, 1, 1, 1], padding = 'SAME'), b_c4))
	conv4 = tf.nn.max_pool(conv4, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
	conv4 = tf.nn.dropout(conv4, keep_prob)

	# 全连接层，4*4*256
	w_d = tf.Variable(tf.random_normal([4 * 4 * 256, 1024], stddev = 0.01))
	b_d = tf.Variable(tf.zeros([1024]))
	dense = tf.reshape(conv4, [-1, w_d.get_shape().as_list()[0]])
	dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
	dense = tf.nn.dropout(dense, keep_prob)

	w_out = tf.Variable(tf.random_normal([1024, category], stddev = 0.01))
	b_out = tf.Variable(tf.zeros([category]))
	out = tf.add(tf.matmul(dense, w_out), b_out)

	return out

def M6_plus():
	# M6+ 40-80-160-320-2fc
	x = tf.reshape(X, shape = [-1, 64, 64, 1])
	# 4 conv layers
	w_c1 = tf.Variable(tf.random_normal([3, 3, 1, 40], stddev = 0.01))
	b_c1 = tf.Variable(tf.zeros([40]))
	conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides = [1, 1, 1, 1], padding = 'SAME'), b_c1))
	conv1 = tf.nn.max_pool(conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

	# 32*32*40
	w_c2 = tf.Variable(tf.random_normal([3, 3, 40, 80], stddev = 0.01))
	b_c2 = tf.Variable(tf.zeros([80]))
	conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides = [1, 1, 1, 1], padding = 'SAME'), b_c2))
	conv2 = tf.nn.max_pool(conv2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

	# 16*16*80
	w_c3 = tf.Variable(tf.random_normal([3, 3, 80, 160], stddev = 0.01))
	b_c3 = tf.Variable(tf.zeros([160]))
	conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides = [1, 1, 1, 1], padding = 'SAME'), b_c3))
	conv3 = tf.nn.max_pool(conv3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
	conv3 = tf.nn.dropout(conv3, keep_prob)

	# 8*8*160
	w_c4 = tf.Variable(tf.random_normal([3, 3, 160, 320], stddev = 0.01))
	b_c4 = tf.Variable(tf.zeros([320]))
	conv4 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv3, w_c4, strides = [1, 1, 1, 1], padding = 'SAME'), b_c4))
	conv4 = tf.nn.max_pool(conv4, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
	conv4 = tf.nn.dropout(conv4, keep_prob)

	# 全连接层，4*4*256
	w_d = tf.Variable(tf.random_normal([4 * 4 * 320, 1024], stddev = 0.01))
	b_d = tf.Variable(tf.zeros([1024]))
	dense = tf.reshape(conv4, [-1, w_d.get_shape().as_list()[0]])
	dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
	dense = tf.nn.dropout(dense, keep_prob)

	w_out = tf.Variable(tf.random_normal([1024, category], stddev = 0.01))
	b_out = tf.Variable(tf.zeros([category]))
	out = tf.add(tf.matmul(dense, w_out), b_out)

	return out


def train_cnn(output):
	# output = ascii_cnn()

	loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits = output, labels = Y))
	optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)

	accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1)), tf.float32))

	# 创建saver用于保存模型
	saver = tf.train.Saver()

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver.restore(sess, "./model/")
		acc = 0.0
		for i in range(2000):
			image_batch, label_batch = getRandomData(train_list, 300)

			sess.run([optimizer, loss],feed_dict = {X: image_batch, Y: label_batch, keep_prob: 0.5})
			print("\r正在训练第 %d 步 ，最近一次测试准确率为%f, 每 5 步更新一次准确率"%(i+1,acc), end="")

			if (i+1)%5==0:
				image_batch, label_batch = getRandomData(test_list, 200)

				acc = accuracy.eval({X: image_batch, Y: label_batch, keep_prob: 1.})

				# print("训练第%d步，准确率为：%f"%(i+1,acc))

			# if (i+1)%200==0:
			# 	saver.save(sess, "./model2/")
		# saver.restore(sess, "./model/")
		print("\n----训练完毕，正在进行十次测试----")
		nums = []
		for i in range(10):
			image_batch, label_batch = getRandomData(test_list, 249)
			acc = accuracy.eval({X: image_batch, Y: label_batch, keep_prob: 1.})
			nums.append(acc)
			print("第%d次测试结果：%f"%(i+1,acc))
		print(nums)
		ac = (np.mean(nums))
		print("测试完毕，平均准确率为：%f" % ac)
		return ac

def test_cnn(output):
	# 创建saver用于保存模型
	saver = tf.train.Saver()
	accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1)), tf.float32))
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		saver.restore(sess, "./model/")

		for i in range(10):
			image_batch, label_batch = getRandomData(test_list, 249)
			acc = accuracy.eval({X: image_batch, Y: label_batch, keep_prob: 1.})
			print("第%d次测试结果：%f"%(i+1,acc))


def tesify_model(output):
	ac_list = []
	for i in range(5):
		print("--------第%d次模型测试---------" % (i+1))
		t1 = time.time()
		ac = train_cnn(output)
		ac_list.append(ac)
		t2 = time.time()
		print("第%d次模型测试 消耗时间: %f" % (i+1, t2 - t1))
	print(ac_list)
	print(np.mean(ac_list))

if __name__ == '__main__':


	Model5_plus = M5_plus()
	train_cnn(Model5_plus)


	# getRandomData(1)
	# getDataBatch(2)
	# convert_pic("G:/Java资料/Code/Python/digital_imgae_processing_design/data_test/test2/0.jpg")
