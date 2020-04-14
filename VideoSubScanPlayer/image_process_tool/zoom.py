# *_*coding:utf-8 *_*
import cv2
import numpy as np

'''将图片的大小拉伸到64*64'''


def magnify(image):
	# 获取到图片的三个通道
	r, c, line = image.shape

	# 将240设为阈值，小于240设置为0，大于240设置为255
	image[image >= 240] = 255
	image[image < 240] = 0
	# 将三个通道合并成一个
	image = image[:, :, 0] & image[:, :, 1] & image[:, :, 2]
	# 将图片进行膨胀
	k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
	image = cv2.dilate(image, k)

	# 扩展宽高
	r1 = r + 10
	c1 = c + 10

	# 计算比例
	row = 64 / r1
	col = 64 / c1

	# 设置全黑图片
	new_img = np.zeros((r1, c1))

	# 相当于在原图加上一周黑边
	new_img[5:r1 - 5, 5:c1 - 5] = image
	# 按计算好的比例等比例先将高度拉伸到64，线性插值
	new_img = cv2.resize(new_img, (int(c1 * row), 64), interpolation = cv2.INTER_LINEAR)

	r2, c2 = new_img.shape
	# 判断高度等比例拉伸后的图片的宽度是大于64还是小于64
	# 若宽度大于64
	if c2 > 64:
		# 计算多出来的像素个数
		c_i = c2 - 64
		# 创建一个更大的正方形的图片
		new_new_img = np.zeros((c2, c2))
		ck = int(c_i / 2)
		# 判断奇偶然后将拉伸高度后的图片放入正方形图片中间位置
		if c_i % 2 != 0:
			new_new_img[ck:c2 - ck - 1, :] = new_img
		else:
			new_new_img[ck:c2 - ck, :] = new_img
		# 再等比例缩小为64*64
		new_new_img = cv2.resize(new_new_img, (64, 64), interpolation = cv2.INTER_AREA)
		return scale(alignCenter(new_new_img),0.75)
	# 同理，若宽度小于64
	else:
		# 计算多出来的像素个数
		c_i = 64 - c2
		new_new_img = np.zeros((64, 64))
		ck = int(c_i / 2)
		# 判断奇偶然后将拉伸高度后的图片放入正方形图片中间位置
		if c_i % 2 != 0:
			new_new_img[:, ck:64 - ck - 1] = new_img
		else:
			new_new_img[:, ck:64 - ck] = new_img

		return scale(alignCenter(new_new_img),0.75)

def getInfo(img):
	row, col = img.shape
	# 获取上下左右间隔信息
	rowLine = np.zeros(row, np.uint8)
	colLine = np.zeros(col, np.uint8)
	# 水平投影 用于确定上下间隔
	for r in range(row):
		for c in range(col):
			if img[r][c] != 0:
				rowLine[r] = 1
				break
	# 水平投影 用于确定左右间隔
	for c in range(col):
		for r in range(row):
			if img[r][c] != 0:
				colLine[c] = 1
				break
	# 计算上下左右黑色间隔
	top, down, left, right = 0, 0, 0, 0
	# print(rowLine)
	# print(colLine)
	for i in rowLine:
		if i == 0:
			top += 1
		else:
			break
	for i in reversed(rowLine):
		if i == 0:
			down += 1
		else:
			break
	for i in colLine:
		if i == 0:
			left += 1
		else:
			break
	for i in reversed(colLine):
		if i == 0:
			right += 1
		else:
			break
	return top,down,left,right


def alignCenter(img):
	# 对创建好的二值图像进行居中处理
	# 灰度读取

	top,down,left,right = getInfo(img)
	# print(top,down,left,right)
	# 移位矩阵
	matShift = np.float32([[1, 0, (right-left)/2], [0, 1, (down-top)/2]])
	# 移位API
	dst = cv2.warpAffine(img, matShift, img.shape)

	return dst

def scale(img,scale_rate):
	# 图像缩放
	img_info = img.shape
	height = img_info[0]
	width = img_info[1]

	# 旋转矩阵,中心点，角度，缩放系数
	matRotate = cv2.getRotationMatrix2D((height * 0.5, width * 0.5), 0, scale_rate)
	ret = cv2.warpAffine(img, matRotate, (height, width))
	# print(scale_rate)
	return ret