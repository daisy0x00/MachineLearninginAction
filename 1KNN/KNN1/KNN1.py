#coding: utf-8

from numpy import *
import operator
from collections import Counter
#import matplotlib
#import matplotlib.pyplotas plt

# 创建数据集和标签
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# KNN算法
def classify0(inX, dataSet, labels, k):
    # 计算欧式距离
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances ** 0.5

    # 选择距离最小的k个点
    sortedDistIndicies = distances.argsort() # .argsort()返回从小到大排序后，原位置对应的索引，而数组并没有排序
    classCount = {}

    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 排序
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]




# 导入特征数据
def file2matrix(filename):
    # 打开文件，得到文件行数
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    # 创建返回的NumPy矩阵
    # 此处创建以零填充的矩阵NumPy，为了简化处理，我们将该矩阵的另一个维度设置为固定值3
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    # 解析文件数据到列表，循环处理文件中的每行数据
    index = 0
    for line in arrayOLines:
        # 使用函数line.strip()截取掉所有的回车字符
        line = line.strip()
        # 使用tab字符将上一步得到的整行数据分割成一个元素列表
        listFromLine = line.split('\t')
        # 取前3个元素，将他们存储到特征矩阵中
        returnMat[index, :] = listFromLine[0 : 3]
        # 利用负索引-1将列表的最后一列存储到向量classLabelVector中
        # 用int()函数通知解释器列表中存储的元素值为整型
        classLabelVector.append(int(listFromLine[-1]))
        index += 1

    return returnMat,classLabelVector

# 归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    # 特征值相除
    normDataSet = normDataSet / tile(ranges, (m , 1))
    return normDataSet, ranges, minVals

# 自包含测试代码，用来测试该分类器的准确率
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs : m, :], datingLabels[numTestVecs : m], 3)
        print("the classifier came back with: %d, the real answer is : %d" %(classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0

    print("the total error rate is %f" %(errorCount / float(numTestVecs)))

# 约会网站预测函数
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles= float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("U will probably like this person:", resultList[classifierResult - 1])