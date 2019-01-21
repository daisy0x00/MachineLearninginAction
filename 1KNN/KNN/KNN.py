#coding:utf-8

# 导入两个模块：科学计算包NumPy和运算符模块operator
from numpy import *
import operator

# 给出训练数据（数据集）以及对应的类别（标签）
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

# KNN算法
# 对未知类别属性的数据集中的每个点依次执行以下操作：
# 1.计算已知类别数据集中的点与当前点之间的距离；
# 2.按照距离递增次序排序
# 3.选取与当前点距离最小的k个点
# 4.确定前k个点所在类别的出现频率
# 5.返回前k个点出现频率最高的类别作为当前点的预测分类
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)

    return sortedClassCount[0][0]