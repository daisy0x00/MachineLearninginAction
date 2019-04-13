#coding:utf-8

import math
from numpy import *
import matplotlib.pyplot as plt

# Logistic回归梯度上升优化算法
# loadDataSet()函数功能：打开文本文件testSet.txt并逐行读取。
# 每行前两个值分别是X1和X2，第三个值是数据对应的类别标签
# 为了方便计算，该函数将X0的值设置为1.0
def loadDataSet():
    # 创建两个空列表
    # dataMat用来存放数据集中的特征X0, X1, X2，其中X0设置为1.0
    # labelMat用来存放每行数据的类别标签
    dataMat = []
    labelMat = []
    # 打开文本文件testSet.txt
    fr = open('testSet.txt')
    # 逐行读取文本文件testSet.txt中的内容
    for line in fr.readlines():
        # 函数strip()用来去掉每行开头和结尾的空白字符（' ', '\n', '\t', '\r'）
        # 函数split()用来切分字符串
        lineArr = line.strip().split()
        # 每行的前两个值分别为X1和X2，为了方便计算，将X0的值设置为1.0
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        # 第三个值是数据对应的类别标签
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

# sigmoid 函数
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

# 梯度上升算法函数
# 参数dataMatIn是一个2维NumPy数组，每列分别代表每个不同的特征，每行则代表每个训练样本。
# 我们现在采用的是100个样本的简单数据集，它包含了两个特征X1和X2，再加上第0维特征X0
# 所以dataMatIn里存放的将是100*3的矩阵
# 参数classLabelss是类别标签，它是一个1*100的行向量
def gradAscent(dataMatIn, classLabels):
    # 获得输入数据并将它们转换为NumPy矩阵
    dataMatrix = mat(dataMatIn)
    # 为了便于矩阵运算，需要将该行向量转换为列向量
    # 做法是将原向量转置，再将它赋值给labelMat
    labelMat = mat(classLabels).transpose()
    # 得到矩阵大小
    m, n = shape(dataMatrix)
    # 变量alpha是向目标移动的步长
    alpha = 0.001
    # maxCycles是迭代次数
    maxCycles = 500

    weights = ones((n, 1))
    for k in range(maxCycles):
        # 预测值
        h = sigmoid(dataMatrix * weights)
        # 计算真实类别与预测类别的插值
        error = (labelMat - h)
        # 按照该差值的方向调整回归系数
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

# 画出数据集和Logistic回归最佳拟合直线的函数
def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    # 列数
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []

    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s = 30, c = 'red', marker = 's')
    ax.scatter(xcord2, ycord2, s = 30, c = 'green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

# 随机梯度上升算法
def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

# 改进的随机梯度上升算法
def stocGradAscent1(dataMatrix, classLabels, numIter = 150):
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

# Logistic回归分类函数
# 函数classifyVector()以回归系数和特征向量作为输入来计算对应的sigmoid值。
# 若sigmoid值大于0.5函数返回1，否则返回0
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

# 函数colicTest()用于打开测试集和训练集，并对数据进行格式化处理
def colicTest():
    # 导入训练集
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
        errorRate =  (float(errorCount) / numTestVec)
        print('the error rate of this test is: %f' %errorRate)
        return errorRate

def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print('after %d iterations the average error rateis: %f' %(numTests, errorSum / float(numTests)))
