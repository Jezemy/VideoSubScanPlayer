# coding=utf-8
import cv2 as cv
import numpy as np
import random
from PIL import Image
import os
"""
在原有数据集的基础上扩充数据集
说明：图片为二值图像，前景为白色，背景为黑色
"""


def dilate(img, num):
	# 输入图像，输出膨胀效果
	# 矩形结构元
	size = (num,num)
	kernel_rect = cv.getStructuringElement(cv.MORPH_RECT, size)
	# 椭圆结构元
	kernel_ellipse = cv.getStructuringElement(cv.MORPH_ELLIPSE, size)
	# 十字形结构元
	kernel_cross = cv.getStructuringElement(cv.MARKER_CROSS, size)

	kernels = [kernel_rect, kernel_ellipse, kernel_cross]

	ret = cv.dilate(img, kernel = random.sample(kernels, 1)[0])
	return ret

def erode(img, num):
	# 输入图像，输出腐蚀效果
	# 矩形结构元
	size = (num, num)
	kernel_rect = cv.getStructuringElement(cv.MORPH_RECT, size)
	# 椭圆结构元
	kernel_ellipse = cv.getStructuringElement(cv.MORPH_ELLIPSE, size)
	# 十字形结构元
	kernel_cross = cv.getStructuringElement(cv.MARKER_CROSS, size)

	kernels = [kernel_rect, kernel_ellipse, kernel_cross]

	ret = cv.erode(img, kernel = random.sample(kernels, 1)[0])
	return ret

def rotate(img):
	# 图像旋转
	img_info = img.shape
	height = img_info[0]
	width = img_info[1]

	angle = random.randint(-5,5)
	# 旋转矩阵,中心点，角度，缩放系数
	matRotate = cv.getRotationMatrix2D((height*0.5, width*0.5), angle, 1)
	ret = cv.warpAffine(img, matRotate, (height, width))
	# print(angle)
	return ret

def noise(img):
	# 添加噪点
	h, w= img.shape
	for i in range(random.randint(5,10)):
		x = random.randint(0,h-1)
		y = random.randint(0,w-1)
		img[x, y] = 255
	return img

def offset(img):
	# 平移
	h, w = img.shape
	x = random.randint(-15,15)
	y = random.randint(-10,10)
	# 移位矩阵
	matShift = np.float32([[1, 0, x], [0, 1, y]])
	# 移位API
	dst = cv.warpAffine(img, matShift, (h, w))
	return dst

def scale(img):
	# 图像缩放
	img_info = img.shape
	height = img_info[0]
	width = img_info[1]

	scale_rate = random.uniform(0.8,1.5)
	# 旋转矩阵,中心点，角度，缩放系数
	matRotate = cv.getRotationMatrix2D((height * 0.5, width * 0.5), 0, scale_rate)
	ret = cv.warpAffine(img, matRotate, (height, width))
	# print(scale_rate)
	return ret

def expand():
	# 腐蚀1-2 膨胀1-4,旋转和加噪点随机
	data_dir_path = "G:\\Java资料\\Code\\Python\\digital_imgae_processing_design\\dataset\\"
	asciiList = [str(index) for index in range(33, 123)]

	# 先腐蚀和膨胀操作，两种操作独立，获得总共21 + 21 *6 = 147张图片
	for numStr in asciiList:
		# 子目录如dataset/a/
		sub_dir = data_dir_path + numStr + "\\"
		# 子目录下的文件名
		sub_dir_list = os.listdir(sub_dir)
		# 因为原来已经有编号20的图片了，编号格式 numStr_index.jpg
		index = 21
		# 腐蚀 参数1-2
		for file_name in sub_dir_list:
			file_dir = sub_dir + file_name
			img = PIL2CV(file_dir)
			for num in [1,2]:
				dst_erode = erode(img,num)
				dst = CV2PIL(dst_erode)
				while os.path.exists(sub_dir+numStr+'_'+str(index)+'.jpg'):
					index += 1
				dst_path = sub_dir+numStr+'_'+str(index)+'.jpg'
				dst.save(dst_path,'jpeg')
				print(dst_path)

		# 膨胀 参数1-4
		for file_name in sub_dir_list:
			file_dir = sub_dir + file_name
			img = PIL2CV(file_dir)
			for num in [1,2,3,4]:
				dst_dilate = dilate(img,num)
				dst = CV2PIL(dst_dilate)
				while os.path.exists(sub_dir+numStr+'_'+str(index)+'.jpg'):
					index += 1
				dst_path = sub_dir+numStr+'_'+str(index)+'.jpg'
				dst.save(dst_path,'jpeg')
				print(dst_path)

	# 对腐蚀膨胀后的照片进行随机旋转
	# 旋转后得到147 + 147  =  294
	for numStr in asciiList:
		# 更新一次目录内的文件
		sub_dir = data_dir_path + numStr + "\\"
		sub_dir_list = os.listdir(sub_dir)
		# 继续预先定好编号
		index = 21

		# 对该子目录下每个文件
		for file_name in sub_dir_list:
			file_dir = sub_dir + file_name
			img = PIL2CV(file_dir)
			# 旋转
			dst_rotate = rotate(img)
			dst = CV2PIL(dst_rotate)
			while os.path.exists(sub_dir+numStr+'_'+str(index)+'.jpg'):
				index += 1
			dst_path = sub_dir+numStr+'_'+str(index)+'.jpg'
			dst.save(dst_path, 'jpeg')
			print(dst_path)

	# 旋转后再加平移 294 + 294 * 2 = 882
	for numStr in asciiList:
		# 更新一次目录内的文件
		sub_dir = data_dir_path + numStr + "\\"
		sub_dir_list = os.listdir(sub_dir)
		# 继续预先定好编号
		index = 21
		# 对该子目录下每个文件
		for i in range(2):
			for file_name in sub_dir_list:
				file_dir = sub_dir + file_name
				img = PIL2CV(file_dir)
				# 平移
				dst_offset = offset(img)
				dst = CV2PIL(dst_offset)
				while os.path.exists(sub_dir + numStr + '_' + str(index) + '.jpg'):
					index += 1
				dst_path = sub_dir + numStr + '_' + str(index) + '.jpg'
				dst.save(dst_path, 'jpeg')
				print(dst_path)

	# 平移后缩放 882 + 882 = 1764
	for numStr in asciiList:
		# 更新一次目录内的文件
		sub_dir = data_dir_path + numStr + "\\"
		sub_dir_list = os.listdir(sub_dir)
		# 继续预先定好编号
		index = 21
		# 对该子目录下每个文件
		for file_name in sub_dir_list:
			file_dir = sub_dir + file_name
			img = PIL2CV(file_dir)
			# 缩放
			dst_offset = scale(img)
			dst = CV2PIL(dst_offset)
			while os.path.exists(sub_dir + numStr + '_' + str(index) + '.jpg'):
				index += 1
			dst_path = sub_dir + numStr + '_' + str(index) + '.jpg'
			dst.save(dst_path, 'jpeg')
			print(dst_path)

	# 缩放后再加噪点 1764 + 1764 = 3528
	for numStr in asciiList:
		# 更新一次目录内的文件
		sub_dir = data_dir_path + numStr + "\\"
		sub_dir_list = os.listdir(sub_dir)
		# 继续预先定好编号
		index = 21
		# 对该子目录下每个文件
		for file_name in sub_dir_list:
			file_dir = sub_dir + file_name
			img = PIL2CV(file_dir)
			# 加噪点
			dst_offset = noise(img)
			dst = CV2PIL(dst_offset)
			while os.path.exists(sub_dir + numStr + '_' + str(index) + '.jpg'):
				index += 1
			dst_path = sub_dir + numStr + '_' + str(index) + '.jpg'
			dst.save(dst_path, 'jpeg')
			print(dst_path)


def PIL2CV(imgPath):
	# 输入图片路径，输出opencv格式
	image = Image.open(imgPath)
	return cv.cvtColor(np.asarray(image), cv.COLOR_RGB2GRAY)

def CV2PIL(image):
	# 输入opencv格式图片，输出Image格式图片
	return Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))


def test():
	image = Image.open("65_1.jpg")
	img = cv.cvtColor(np.asarray(image), cv.COLOR_RGB2GRAY)
	img = scale(img)
	cv.imshow('img',img)
	cv.waitKey(0)

if __name__ == '__main__':
	expand()
	# test()