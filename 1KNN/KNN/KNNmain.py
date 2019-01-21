#coding:utf-8

import KNN
from numpy import *

dataSet,labels = KNN.createDataSet()
input = array([0.1, 0.3])
k = 3
output = KNN.classify0(input, dataSet, labels, k)
print("测试数据：", input, "分类结果： ", output)