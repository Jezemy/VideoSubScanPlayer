# *_*coding:utf-8 *_*
import cv2
import time
import cut
import cutFont as cf
import numpy as np


def SubText_Detection(frame1, frame2):
	# 设置ROI和阈值分割
	frame1 = cut.set_ROI(frame1)
	frame1 = cf.threshold(frame1, 240)

	# 设置ROI和阈值分割
	frame2 = cut.set_ROI(frame2)
	frame2 = cf.threshold(frame2, 240)

	# 计算前后两帧的方差
	r, c = frame1.shape
	img = frame2 - frame1
	img[img <= 0] = 0

	sum_r = np.sum(img ** 2, axis = 1)
	sum = np.sum(sum_r, axis = 0)

	e = sum / (r * c) * 100

	# 若方差大于0.3说明出现字幕
	if e >= 0.1:
		# 如果大于0.3说明有新字幕出现，返回True
		return True
	else:
		return False
