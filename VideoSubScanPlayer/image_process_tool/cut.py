# *_*coding:utf-8 *_*
import cv2
import numpy as np
import cutFont as cf
import zoom


def Nomalize(array, value):
	row, col = array.shape
	for i in range(row):
		for j in range(col):
			if array[i][j] > value:
				array[i][j] = 1
			else:
				array[i][j] = 0
	return array


'''设置视频字幕的范围区域'''


def set_ROI(image):
	r, c, line = image.shape
	return image[int(r * 0.74):r, int(c / 10):int(c / 2) * 2, :]


'''运行分割算法,origineImage表示原图，i表示第几帧的图片'''


def Image_Division(origineImage):
	# 先将彩色图片保存
	rgb_img = origineImage.copy()

	# 设置字幕的区域范围
	origineImage = set_ROI(origineImage)
	rgb_img = set_ROI(rgb_img)

	# 进行字符切割
	Position, B_Position, img = cf.cutFont(origineImage, 1, 0)

	# 如果Position为空则说明没有字幕,直接返回
	if Position == None:
		return False, None, None

	# 用于存放适用于卷积神经网络识别的图片格式
	matrix = []

	# 根据确定的位置分割字符
	length = len(Position)
	# 用于存储最终分割好的所有字幕的像素索引
	final_Position = []

	for m in range(length):
		x1 = Position[m][0]
		y1 = Position[m][1]
		x2 = Position[m][2]
		y2 = Position[m][3]

		# 如果字符的长度大于15个像素，就可能存在两个字符连在一起的情况，这种情况做二次分割
		if x2 - x1 >= 15:
			# new_img = origineImage[y1:y2, x1:x2]

			# 二次分割
			sub_Position, sub_B_Position, sub_img = cf.cutFont(origineImage, 2, [x1, x2, y1, y2])

			# 若sub_Position为空则说明没有字幕，直接返回
			if sub_Position == None:
				return False, None, None

			# 保存二次分割的字符
			for n in range(len(sub_Position)):
				xx1 = sub_Position[n][0]
				yy1 = sub_Position[n][1]
				xx2 = sub_Position[n][2]
				yy2 = sub_Position[n][3]
				# 获得二次分割后的图片
				img_i = rgb_img[yy1:yy2, xx1:xx2]
				# 将字体拉伸到64*64
				new_img_i = zoom.magnify(img_i)

				matrix.append(Nomalize(new_img_i,128).reshape((4096)))
				# 添加到最终位置列表里
				final_Position.append(sub_Position[n])



		# 小于15个像素直接保存
		else:
			# 添加到最终位置列表里
			final_Position.append(Position[m])
			img_j = rgb_img[y1:y2, x1:x2]
			# 拉伸64*64
			new_img_j = zoom.magnify(img_j)
			matrix.append(Nomalize(new_img_j,128).reshape((4096)))

	# 用来存储空格位于这一整句话中的索引
	blank_index = []
	count = 0

	# 遍历空格的个数
	for blank in range(len(B_Position)):

		# 遍历每一个字符
		for char in range(count, len(final_Position) - 1):

			# 若空格的起始位置大于第char个字符的终止位置并且小于第char+1个字符的起始位置，则可以认为这个空格存在于第char和第char+1个字符之间
			if (B_Position[blank][0] >= final_Position[char][2]) and (
					B_Position[blank][2] <= final_Position[char + 1][0]):
				# 添加到列表中
				blank_index.append(char)
				# 重新定义起始位置
				count = char + 1
				break

	return True, np.array(matrix, dtype = np.uint8), blank_index


if __name__ == '__main__':
	img = cv2.imread('D:/video1/test26.jpg')
	a, b, c = Image_Division(img)
	# print(b)
