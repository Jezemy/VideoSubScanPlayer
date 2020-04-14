from PIL import Image, ImageDraw, ImageFont
import os
import cv2 as cv
import numpy as np

"""
根据font文件夹中的字体生成数据集
"""

# 根据字体生成指定字符
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

def start_to_make_data():
	dir = 'font/'
	# 获取字体文件名
	font_name = os.listdir(dir)
	# 获取完整字体路径
	font_list = [os.path.join(dir, file) for file in font_name]  # if file[-3:] == "bin"]
	# 指定要构造的字符，ascii 32到126
	asciiList = [index for index in range(33, 123)]
	# print(strList)
	# print(font_name)
	makeImg(font_list, asciiList)

def makeImg(fontList, asciiList):
	# 64 * 64 :
	width = 64
	height = 64
	for ascii_id in asciiList:
		# 创建文件夹
		path = os.path.abspath('.')
		if os.path.exists(path + '\\' + 'dataset\\' + str(ascii_id)) == False:
			os.mkdir(path + '\\' + 'dataset\\' + str(ascii_id))

		# 根据字体列表创建数据
		for index, fontName in enumerate(fontList):
			image = Image.new('RGB', (width, height), (0, 0, 0))
			# 创建Font对象:
			font = ImageFont.truetype(fontName, 48)
			# 创建Draw对象:
			draw = ImageDraw.Draw(image)
			# 输出文字
			position =(8,2)
			draw.text(position, chr(ascii_id), font=font, fill="#FFFFFF", spacing=0, align='left')
			# 构建文件保存目录并输出
			file_name = path + '\\' + 'dataset\\' + str(ascii_id) + '\\'+ str(ascii_id) + "_" + str(index) + '.jpg'
			print(file_name)
			# 居中图片并保存
			dst = alignCenter(image)
			dst.save(file_name, 'jpeg')


def alignCenter(photo):
	# 对创建好的二值图像进行居中处理
	# 灰度读取
	# row, col = image.size
	# print(row,col)
	image = cv.cvtColor(np.asarray(photo), cv.COLOR_RGB2GRAY)

	# 临时放大方便调试
	# img = cv.resize(img,(200,200))
	th, img = cv.threshold(image,128,255,cv.THRESH_BINARY)


	top,down,left,right = getInfo(img)
	# print(top,down,left,right)
	# 移位矩阵
	matShift = np.float32([[1, 0, (right-left)/2], [0, 1, (down-top)/2]])
	# 移位API
	dst = cv.warpAffine(image, matShift, img.shape)

	return Image.fromarray(cv.cvtColor(dst,cv.COLOR_BGR2RGB))


if __name__ == '__main__':
	start_to_make_data()
	# alignCenter('test.jpg')


