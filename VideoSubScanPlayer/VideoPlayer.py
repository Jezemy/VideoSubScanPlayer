# coding=utf-8
'''
VideoPlayer 主程序
'''
import sys

# 把当前位置作为搜索路径以导包
sys.path.append('image_process_tool\\')

from image_process_tool.readVideo import SubText_Detection
from image_process_tool.cut import Image_Division
from baidu_translator import Translation
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import tensorflow as tf
import traceback
import cv2 as cv
import vthread
import threading


def CNN_Model():
    # M4+模型流图
    x = tf.reshape(X, shape=[-1, 64, 64, 1])
    # print(x,x.graph)
    # 3 conv layers
    w_c1 = tf.Variable(tf.compat.v1.random_normal([3, 3, 1, 64], stddev=0.01))
    b_c1 = tf.Variable(tf.compat.v1.zeros([64]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 32*32*64
    w_c2 = tf.Variable(tf.compat.v1.random_normal([3, 3, 64, 128], stddev=0.01))
    b_c2 = tf.Variable(tf.compat.v1.zeros([128]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 全连接层，16*16*128
    w_d = tf.Variable(tf.compat.v1.random_normal([16 * 16 * 128, 1024], stddev=0.01))
    b_d = tf.Variable(tf.compat.v1.zeros([1024]))
    dense = tf.reshape(conv2, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.compat.v1.Variable(tf.compat.v1.random_normal([1024, category], stddev=0.01))
    b_out = tf.compat.v1.Variable(tf.compat.v1.zeros([category]))
    out = tf.add(tf.matmul(dense, w_out), b_out)

    return out


# 获取默认图，后续操作都在这个图下操作
graph = tf.compat.v1.get_default_graph()
with graph.as_default():
    # 初始化变量
    X = tf.compat.v1.placeholder(tf.float32, [None, 64 * 64])
    keep_prob = tf.compat.v1.placeholder(tf.float32)
    category = 90
    sess = tf.compat.v1.Session()
    out = CNN_Model()
    # 获取训练好的模型
    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, './model/')


def Recognization(Division_Pic_npArray):
    # 在创建出来的图下进行识别操作，最后返回各个识别结果的ascii字符
    with graph.as_default():
        result = tf.argmax(sess.run(out, feed_dict={X: Division_Pic_npArray, keep_prob: 0.5}), 1).eval(session=sess)
        return [chr(i + 33) for i in result]


class VideoPlayer(QMainWindow):
    def __init__(self):
        super(VideoPlayer, self).__init__()

        # 线程队列
        self.ThreadList = []

        # 字幕队列
        self.subtext_Dict = {}

        # 播放状态，0表示没有视频在播放，1表示有文件播放但是暂停了，2表示正在播放
        self.STATUS = 0

        # 视频播放地址
        self.videoUrl = None

        # 设置定时器
        self.timer = None
        # 设置播放器
        self.video = None
        # 总帧数
        self.totalFrameNum = None
        # 设置帧数
        self.fps = None
        # 设置每秒等待时间
        self.waitTime = None

        # 初始化
        self.initUI()
        self.center()

    def initUI(self):
        # 跟UI有关的所有的内容
        self.setWindowTitle("VideoPlayer")
        self.resize(1280, 720)
        self.setWindowIcon(QIcon('icon/video.ico'))

        # 设置界面布局
        Layout = QVBoxLayout()
        layout_button = QHBoxLayout()

        # 设置视频显示的组件img_label
        self.img_label = QLabel("视频")
        self.img_label.setScaledContents(True)
        self.img_label.setAlignment(Qt.AlignCenter)
        Img = QImage('1.jpg').scaled(1280, 720)
        pixImg = QPixmap.fromImage(Img)
        self.img_label.setPixmap(pixImg)
        self.img_label.resize(1280, 720)
        Layout.addWidget(self.img_label)

        # 设置字幕显示的组件subtext_label
        self.subtext_label = QLineEdit("字幕")
        self.subtext_label.setAlignment(Qt.AlignCenter)
        self.subtext_label.setFont(QFont('Arial', 18))
        self.subtext_label.setStyleSheet('background - color: rgb(0, 0, 0)')
        self.subtext_label.adjustSize()
        Layout.addWidget(self.subtext_label)

        # 按钮布局
        self.btn_open = QPushButton("打开")
        self.btn_open.setIcon(QIcon('icon/open_button.ico'))
        self.btn_open.clicked.connect(self.open_button)
        self.btn_open.clicked.connect(self.statusChange)

        self.btn_play = QPushButton("播放")
        self.btn_play.setIcon(QIcon('icon/play_button.ico'))
        self.btn_play.clicked.connect(self.play_button)
        self.btn_play.clicked.connect(self.statusChange)

        self.btn_pause = QPushButton("暂停")
        self.btn_pause.setIcon(QIcon('icon/stop_button.ico'))
        self.btn_pause.clicked.connect(self.pause_button)
        self.btn_pause.clicked.connect(self.statusChange)

        self.btn_extractSubtitle = QPushButton("导出字幕")
        self.btn_extractSubtitle.setIcon(QIcon("icon/btn_extractSubtitle.ico"))
        self.btn_extractSubtitle.clicked.connect(self.extractSubtitle)

        self.statusBar().showMessage("打开视频开始播放")

        # 将创建好的按钮添加进按钮布局
        layout_button.addWidget(self.btn_open)
        layout_button.addWidget(self.btn_play)
        layout_button.addWidget(self.btn_pause)
        layout_button.addWidget(self.btn_extractSubtitle)
        Layout.addLayout(layout_button)

        # 将创建好的布局加载进主程序
        mainFrame = QWidget()
        mainFrame.setLayout(Layout)
        self.setCentralWidget(mainFrame)

    def statusChange(self):
        # 当按下打开，播放，暂停，三个按钮的时候触发
        # 状态为0表示之前没有任何文件打开
        if self.STATUS == 0:
            # 若字幕数量太少就先缓冲
            if len(self.subtext_Dict) < 5:
                self.statusBar().showMessage("缓冲中....,请稍后重试", 5000)
                return None
            self.statusBar().showMessage("打开视频或开始播放", 5000)
        elif self.STATUS == 1:
            self.statusBar().showMessage("视频已暂停.....", 5000)
        elif self.STATUS == 2:
            self.statusBar().showMessage("视频已播放.....", 5000)

    def open_button(self):
        # 打开按钮的绑定事件
        image_file, _ = QFileDialog.getOpenFileName(self, '打开视频', '', '视频文件 (*.mp4 *.mov *.avi *.mkv)')
        if image_file == "":
            # 如果触发后又没选择文件就返回
            return None
        else:
            # 判断是否是第二次打开文件，是就进行重设reset
            if self.STATUS > 0:
                self.reset()
            print(image_file)
            # 根据读入的文件初始化操作
            self.videoUrl = image_file
            # 设置计时器
            self.timer = QTimer()
            # 通过opencv读取视频
            self.video = cv.VideoCapture(self.videoUrl)
            # 设置总帧数，帧数率，计算计时器每次等待时间和启用函数
            self.totalFrameNum = self.video.get(7)
            self.fps = self.video.get(cv.CAP_PROP_FPS)
            self.waitTime = int((1 / self.fps) * 1000)
            self.timer.timeout.connect(self.show_image)
            # 更新当前的状态
            self.STATUS = 1
            # 创建子进程进行图像字幕检测分割识别操作
            ImgPro = Img_Process(self.videoUrl, self.subtext_Dict)
            self.ThreadList.append(ImgPro)
            ImgPro.start()

    def show_image(self):
        # 此函数由timer计时器调用
        # 从视频获取一帧图像
        success, frame = self.video.read()
        # 获取下一个位置索引
        index = self.video.get(cv.CAP_PROP_POS_FRAMES)

        # 如果读取成功
        if success:
            # 获取长宽
            height, width = frame.shape[:2]
            # opencv读取的图像通道是BGR，因此需要转为RGB
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            # 将获取到的图片放到面板上
            Image = QImage(rgb[:], width, height, QImage.Format_RGB888)
            pixImg = QPixmap.fromImage(Image.scaled(1280, 720))
            self.img_label.setPixmap(pixImg)

            # 判断是否存在字幕，如果当前帧存在字幕，就将字幕显示出来
            if index in self.subtext_Dict:
                self.subtext_label.setText(self.subtext_Dict[index])

        else:
            # 播放失败或播放完毕
            self.reset()

    def play_button(self):
        # 播放按钮绑定事件
        # 如果视频捕捉器，计时器，视频地址有不存在的，表示还没有任何视频加载，直接调用一次打开按钮的绑定事件
        if self.video == None or self.timer == None or self.videoUrl == None:
            self.open_button()

        # 如果状态为0，表示当前没有播放任何文件
        if self.STATUS == 0:
            # 如果字幕数量不够多，就先等等
            if len(self.subtext_Dict) < 5:
                self.statusBar().showMessage("缓冲中....,请稍后重试", 5000)
                return None

            # 字幕数量够多了就可以播放了
            self.timer.start(self.waitTime)
            self.STATUS = 2

        # 如果状态为1 表示是暂停状态，把计时器开启，再更新状态
        elif self.STATUS == 1:
            self.timer.start(self.waitTime)
            self.STATUS = 2

        # 其他情况比如状态为2的状态，不予理会，因为本来就在播放状态
        else:
            return None

    def pause_button(self):
        # 暂停按钮的绑定函数
        # 如果当前在播放状态，就将计时器停止，并更新状态。
        if self.STATUS == 2:
            self.timer.stop()
            self.STATUS = 1

        # 其他情况不予理会
        else:
            return None

    def extractSubtitle(self):

        # 判断字幕处理进程是否完全结束
        for thread in self.ThreadList:
            if not thread.isDone():
                QMessageBox.about(self, '提示', '请等待字幕全部加载完毕')
                return None

        # 设置保存位置
        filename, _ = QFileDialog.getSaveFileName(self, '保存文件', '')
        if filename == "":
            # 如果触发后又没选择文件就返回
            return None
        file_path = filename if (filename[-4:] == '.srt') else filename + '.srt'
        # print(file_path)

        # 打开文件开始进行写入
        with open(file_path, 'w') as f:
            # 先获取字幕字典的所有key,sorted对字典操作，返回的是所有排好序的key的列表
            sort_subtext_key = sorted(self.subtext_Dict)
            # 记录开始时间，开始位置，结束时间
            time_start = None
            time_start_key = None
            time_end = None
            index = -1

            # 对字幕字典进行遍历，每两个字幕确定一段时间
            for key in sort_subtext_key:
                index += 1
                # print(key)
                # 如果刚开始为None，说明是第一个字幕
                if time_start == None or time_start_key == None:
                    print("if")
                    # 第一个字幕就先赋值给time_start
                    time_start = self.formatTime(key)
                    time_start_key = key
                    continue

                # 按srt字幕格式进行写入
                time_end = self.formatTime(key)
                f.write("%d\n" % index)
                f.write("%s --> %s\n" % (time_start, time_end))
                f.write("%s\n" % self.subtext_Dict[time_start_key])
                f.write("\n")
                # print(index)
                time_start_key = key
                time_start = time_end

            # 添加最后一个字幕
            time_final = self.formatTime(self.totalFrameNum)
            f.write("%d\n" % (len(self.subtext_Dict) + 1))
            f.write("%s --> %s\n" % (time_end, time_final))
            f.write("%s\n" % self.subtext_Dict[sort_subtext_key[-1]])
            f.write("\n")
        QMessageBox.about(self, '操作成功', '成功导出srt字幕')

    def formatTime(self, frameNum):
        # 输入帧数，格式化为H:M:S的格式，用于字幕制作
        # 时间= 帧数/帧数率，单位是s
        t = frameNum / self.fps
        H, M, S, s = 0, 0, 0, 0
        # 获取H的大小
        while t >= 3600:
            t -= 3600
            H += 1

        # 获取M的大小
        while t >= 60:
            t -= 60
            M += 1

        # 获取S和s的大小
        S = int(t)
        s = int((t - S) * 1000)

        # 格式化返回
        return "%02d:%02d:%02d,%03d" % (H, M, S, s)

    def reset(self):
        if self.timer is not None:
            self.timer.stop()
        if self.video is not None:
            self.video.release()
        self.timer = None
        self.video = None
        self.STATUS = 0
        self.subtext_Dict = {}
        for thread in self.ThreadList:
            thread.close()

    def center(self):
        # 让窗口居中
        # (屏幕长度-窗口长度)/2 得到长度坐标，高度同理
        screen = QDesktopWidget().screenGeometry()
        # 获取窗口坐标系
        size = self.geometry()
        # 计算新坐标位置
        newLeft = (screen.width() - size.width()) / 2
        newTop = (screen.height() - size.height()) / 3

        self.move(int(newLeft), int(newTop))


class Img_Process(threading.Thread):
    # 用于处理指定视频的字幕检测，字幕分割，字幕识别功能
    def __init__(self, videoUrl, Queue: dict):
        # 传入videoUrl为视频地址，Queue字典为字母存放字典
        super().__init__()
        # 临界资源index，表示当前检测帧的位置
        self.Index = 0
        # 通过opencv获取视频检测器
        self.video = cv.VideoCapture(videoUrl)
        self.video.set(cv.CAP_PROP_POS_FRAMES, self.Index)
        # 获取总帧数
        self.totalFrameNum = self.video.get(7)
        self.Queue_subtext = Queue
        # 设置帧之间字幕检测的帧步长
        self.step = 5
        # 设置运行状态，True表示可以运行，False表示不能运行
        self.RunStatus = True
        # 设置是否处理完
        self.isAllDone = False

    @vthread.thread(5)
    def run(self):
        # 主函数，使用5个进程加速处理
        # 无限循环直到处理完所有图片，除非状态为不可运行
        while True and self.RunStatus:
            # 获取视频帧数
            success, frame_start, frame_end, index = self.getFrame()

            # 如果返回为None 说明视频已经看完了
            if not success:
                break

            # 检测是否变换
            isChanged = SubText_Detection(frame_start, frame_end)
            # print(index, " ", index + 5, " ", isChanged)

            # 没有改变的话就不用进行下一步操作
            if not isChanged:
                continue

            # 进行图像分割
            success, Division_Pic_npArray, blank_index_list = Image_Division(frame_end)

            # 说明这里没字幕
            if not success:
                self.Queue_subtext[index + 5] = "   "
                continue

            # 识别
            Recognized_Str_list = Recognization(Division_Pic_npArray)

            # 字符串加空格
            Final_Str = self.StrAddBlank(Recognized_Str_list, blank_index_list)

            # 调用百度翻译API进行翻译
            # Translated_Str = Translation(Final_Str)

            # 将翻译好的字幕添加进字幕字典里
            self.Queue_subtext[index + 5] = Final_Str

            # print(blank_index_list)
            print(Final_Str)
            print(len(Final_Str))
        # print(Translated_Str)

        # 设置状态
        self.isAllDone = True
        print("-----Thread Quit-----")

    def close(self):
        # 通过设置不可运行，使得run函数开启的线程主动退出
        # 避免强制结束进程而造成的异常
        self.RunStatus = False

    def isDone(self):
        return self.isAllDone

    @vthread.atom
    def getFrame(self):
        # 对临界资源index的相关操作，已加上锁，同时只允许一个线程访问此函数

        # 超过视频帧数大小，说明结束
        if self.Index + 5 > self.totalFrameNum:
            return False, None, None, None

        # 取当前位置帧为帧起点
        self.video.set(cv.CAP_PROP_POS_FRAMES, self.Index)
        success, frame_start = self.video.read()

        # 设置当前位置增加一倍步长
        self.Index += self.step
        self.video.set(cv.CAP_PROP_POS_FRAMES, self.Index)

        # 取增加步长后的位置帧为帧终点
        success, frame_end = self.video.read()

        return True, frame_start, frame_end, self.Index - self.step

    def StrAddBlank(self, Recognized_Str_list: list, blank_index_list: list):
        # 为识别后的字符列表填充空格，返回字符串
        for index in range(len(blank_index_list)):
            Recognized_Str_list.insert(blank_index_list[index] + 1 + (index), " ")
        return ("".join(Recognized_Str_list))


if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        main = VideoPlayer()
        main.show()
        sys.exit(app.exec_())
    except Exception:
        traceback.print_exc()

