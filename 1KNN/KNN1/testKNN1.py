#coding:utf-8


import KNN1
import matplotlib
import matplotlib.pyplot as plt
from numpy import *
from matplotlib.font_manager import FontProperties
import numpy as np

# 使用file2matrix函数读取文件数据，检查数据内容
#datingDataMat, datingLabels = KNN1.file2matrix('datingTestSet2.txt')
#print(datingDataMat)
#print("\n***************************************************************************************\n")
#print(datingLabels[0:20])

# 分析数据，使用Matplotlib创建散点图
## case1:没有样本类别标签的约会数据散点图
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.scatter(datingDataMat[: , 1], datingDataMat[: , 2])
#plt.show()


## case2:采用色彩来标记不同样本分类，以便更好的理解数据信息
## Matplotlib库提供的scatter函数支持个性化标记散点图上的点
#fig = plt.figure()
#ax1 = fig.add_subplot(111)
#ax1.scatter(datingDataMat[: , 1], datingDataMat[: , 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
#plt.show()

## case3:采用矩阵的第一列和第二列属性展示数据
#font = FontProperties(fname = r"C:\Windows\Fonts\SimSun.ttc", size = 14)
#group, labels = KNN1.createDataSet()
#fig = plt.figure()
#ax2 = fig.add_subplot(111)
#datingDataMat, datingLabels = KNN1.file2matrix("datingTestSet2.txt")
#datingLabels = np.array(datingLabels)
#idx_1 = np.where(datingLabels == 1)
#p1 = ax2.scatter(datingDataMat[idx_1, 0], datingDataMat[idx_1, 1], marker = '*', color = 'r', label = '1', s = 10)
#idx_2 = np.where(datingLabels == 2)
#p2 = ax2.scatter(datingDataMat[idx_2, 0], datingDataMat[idx_2, 1], marker = 'o', color = 'g', label = '2', s = 20)
#idx_3 = np.where(datingLabels == 3)
#p3 = ax2.scatter(datingDataMat[idx_3, 0], datingDataMat[idx_3, 1], marker = '+', color = 'b', label = '3', s = 30)
#plt.xlabel("每年获取的飞行里程数", fontproperties = font)
#plt.ylabel("玩视频游戏所消耗的时间百分比", fontproperties = font)
#ax2.legend((p1, p2, p3), ("不喜欢", "魅力一般", "极具魅力"), loc = 2, prop = font)
#plt.show()

# 测试归一化特征值函数
#datingDataMat, datingLabels = KNN1.file2matrix("datingTestSet2.txt")
#normMat, ranges, minVals = KNN1.autoNorm(datingDataMat)
#print(normMat)
#print("\n***************************************************************************************\n")
#print(ranges)
#print("\n***************************************************************************************\n")
#print(minVals)

# 测试KNN1中自包含的测试函数
#print(KNN1.datingClassTest())

# 测试约会网站预测函数
KNN1.classifyPerson()
