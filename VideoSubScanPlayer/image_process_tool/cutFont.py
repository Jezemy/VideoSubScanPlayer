# *_*coding:utf-8 *_*
import cv2
import numpy as np



'''水平投影'''


def getHProjection(image):

    return np.sum(image/255, axis=1)

'''垂直投影'''


def getVProjection(image):


    return np.sum(image/255, axis=0)

#阈值分割
def threshold(image, hold):
    image[image > hold] = 255
    image[image <= hold] = 0
    #将三个通道合并成一个通道
    return image[:, :, 0] & image[:, :, 1] & image[:, :, 2]

#index为第几次分割
def cutFont(origineImage, index, row_col):


    #将彩色图片保存
    rgb_img = origineImage.copy()

    #阈值分割
    if index == 1:
        image = threshold(origineImage, 240)
    else:
        image = threshold(origineImage, 230)

    #膨胀操作
    if index == 1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
        image = cv2.dilate(image, k)
    else:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        image = cv2.dilate(image, k)




    # 图像高与宽
    (h, w) = image.shape

    #print(w)
    Position = []
    B_Position = []
    H_Start = []
    H_End = []

    if index == 1:

        # 水平投影
        H = getHProjection(image)

        # 若没有字幕像素点
        if np.sum(H, axis=0) == 0:
            return None, None, None

        start = 0




        #确定行的起点和终点位置
        for i in range(len(H)):
            if H[i] > 0 and start == 0:
                H_Start.append(i)
                start = 1
            elif H[i] <= 0 and start == 1:
                H_End.append(i)
                start = 0

        # 如果白色字碰到了边缘导致没有结尾，则直接添加最后一个像素点作为结尾
        if (len(H_Start) > len(H_End)):
            H_End.append(len(H))

    else:
        #若为二次分割，行已经确定好了，直接添加到两个列表中
        H_Start.append(row_col[2])
        H_End.append(row_col[3])



    #判断是否为空格的条件，计算空格的长度
    count = 0



    for i in range(len(H_Start)):

        # 获取行图像
        cropImg = image[H_Start[i]:H_End[i], 0:w]
        # 对行图像进行垂直投影
        W = getVProjection(cropImg)
        #是否找到一个字幕起始位置,1为找到，0为未找到
        Wstart = 0
        #是否找到一个字幕终止位置，1为找到，0为未找到
        Wend = 0
        #字幕起始位置的索引
        W_Start = 0
        #字幕终止位置的索引
        W_End = 0
        #是否找到一个空格的起始位置，1为找到，0为未找到
        Bstart = 0
        #是否找到一个空格的终止位置，1为找到，0为未找到
        Bend = 0
        #空格起始位置索引
        B_Start = 0
        #空格终止位置索引
        B_End = 0

        if index == 1:
            #第一次分割就列的起始位置开始切割
            c_start = 1
            c_end = len(W) - 1
        else:
            #第二次分割已经确定了分割列的位置，可以直接指定
            c_start = row_col[0]
            c_end = row_col[1]

        #对列投影后的列表进行遍历，找出字符和空格的位置
        for j in range(c_start-1, c_end + 1):
            #若白色像素个数大于0并且未找到字符的开头位置
            if W[j] > 0 and Wstart == 0:
                #将j赋值个字符开头索引
                W_Start = j
                #找到开头
                Wstart = 1
                #未找到结尾
                Wend = 0
            #若白色像素个数小于等于0并且已经找到开头
            if W[j] <= 0 and Wstart == 1:
                #将j赋值个字符结束索引
                W_End = j
                #未找到开头
                Wstart = 0
                #找到结尾
                Wend = 1
            #若白色像素个数小于等于0并且未找到空格开头并且是第一次分割
            if W[j] <= 0 and Bstart == 0 and index == 1:
                #将j赋值给空格起始索引
                B_Start = j
                #找到空格开头
                Bstart = 1
                #未找到空格结尾
                Bend = 0

            #若白色像素个数大于0并且已经找到空格的开头位置并且是第一次分割
            if W[j] > 0 and Bstart == 1 and index == 1:
                #将j赋值给空格结束索引
                B_End = j
                #未找到空格开头
                Bstart = 0
                #找到空格结尾
                Bend = 1

            #如果白色像素个数小于0并且已经找到了空格开头并且是第一次分割，就将count自增来计算这个空格的长度
            if W[j] <= 0 and Bstart == 1 and index == 1:
                count += 1

            #若找到空格的结尾了
            if Wend == 1:
                #将切割出来的字符存放到Position中
                Position.append([W_Start, H_Start[i], W_End, H_End[i]])
                #将结尾设为0重新寻找
                Wend = 0

            #找到空格结尾并且这个空格的长度是大于7个像素小于100个像素并且是第一次分割
            if Bend == 1 and count >= 7 and count < 100 and index == 1:
                #将空格添加到B_Position中
                B_Position.append([B_Start, H_Start[i], B_End, H_End[i]])
                #长度置为0重新寻找
                count = 0
            #若空格的长度不符合要求
            if Bend == 1 and (count < 7 or count >= 100) and index == 1:
                #不存储此空格并直接将长度重新设为0
                count = 0


    return Position, B_Position, image
