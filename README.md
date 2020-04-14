# 自动翻译字幕的播放器

![效果图](https://github.com/Jezemy/VideoSubScanPlayer/blob/master/pic/display01.png?raw=true)


# 简单介绍
本项目作为本人的数字图像处理课程设计，是一个简陋的视频内嵌字幕识别播放器。

通过数字图像处理技术处理视频内嵌的字幕，然后通过卷积神经网络进行识别，并利用百度翻译接口进行翻译。

UI界面使用的是PYQT，深度学习框架是TensorFlow

图形处理主要使用OpenCV，多线程库使用的是vthread

# 支持功能
- 支持导入mp4，mov，avi，mkv格式的视频
- 支持处理视频下方的白色字幕
- 支持将字幕导出

# 运行环境需求
- TensorFlow 2.0 (原版本是1.4版本，已将主要代码改成兼容1的代码，tf.compat.v1.xxx模式)
- OpenCV
- Pyqt
- Numpy
- VThread
- PIL
- Retrying

# 目录说明
### 文件夹
- image_process_tool 提供了分割视频字幕的工具
- model  训练好的模型
- icon 程序图标
- font 数据集字体
- video Test 用于测试的视频文件

### 文件
- baidu_translator.py 百度翻译API调用
- CNN_new.py 训练CNN模型
- font_data.py 根据字体font生成数据集
- expand_dataset.py 在生成的数据集基础上进行泛化
- VideoPlayer.py 视频播放器主程序


# 使用说明
安装好依赖库后，在VideoSubScanPlayer文件夹下运行代码
```python
python VideoPlayer.py
```

# 开启翻译的方式
- 首先要有一个百度翻译接口，没有的话先[注册](https://api.fanyi.baidu.com/)
- 在开发者信息中得到App ID和密钥
- 修改baidu_translator.py中的部分代码

![修改1](https://github.com/Jezemy/VideoSubScanPlayer/blob/master/pic/instruction1.png?raw=true)

- 修改VideoPlayer.py中的部分代码

![修改2](https://github.com/Jezemy/VideoSubScanPlayer/blob/master/pic/instruction2.png?raw=true)

# 翻译效果图
![translated01.jpg](https://github.com/Jezemy/VideoSubScanPlayer/blob/master/pic/translated01.jpg?raw=true)

![translated02.jpg](https://github.com/Jezemy/VideoSubScanPlayer/blob/master/pic/translated02.jpg?raw=true)

# 不足与缺陷
- 由于图像处理算法的问题，因此对原视频字幕要求比较高，最好是字体清晰的白字黑底字幕，且字体间距不能太靠近。后续可能会改用其他算法。
- 暂时不支持拖动，只能从头播放到尾，然后通过导出字幕的方式保留字幕信息。
- 需要使用百度翻译接口才可以翻译

# 附预览图
### 模块分布流程图
![process01.jpg](https://github.com/Jezemy/VideoSubScanPlayer/blob/master/pic/process01.jpg?raw=true)

### 图像处理流程图
![process02.jpg](https://github.com/Jezemy/VideoSubScanPlayer/blob/master/pic/process02.jpg?raw=true)

### 多线程处理流程图
![process03.jpg](https://github.com/Jezemy/VideoSubScanPlayer/blob/master/pic/process03.jpg?raw=true)