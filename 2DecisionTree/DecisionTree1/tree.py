#coding:utf-8

from math import log
import operator

# 计算给定数据集的香农熵
def calcShannonEnt(dataSet):
    # 计算数据集中实例的总数
    numEntries = len(dataSet)
    # 创建一个数据字典，它的键值是数据集的最后一列的数值
    labelCounts = {}
    # 遍历数据集，为所有可能的分类创建字典
    for featVec in dataSet:
        # 一般数据集的每个元素也是一个列表，这个列表元素的最后一个元素表示该元素所属类别
        currentLabel = featVec[-1]
        # 如果当前键值不存在，则扩展字典并将当前键值加入字典，每个键值都记录了当前类别出现的次数
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    # 计算香农熵
    shannonEnt = 0.0
    # 使用所有类标签的发生频率计算类别出现的概率，并用这个概率计算香农熵
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)

    return shannonEnt

# 创建数据集
def createDataset():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']
               ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

# 按照给定特征划分数据集
# 当我们按照某个特征划分数据集时，就需要将所有符合要求的元素抽取出来
# dataSet: 待划分的数据集 axis： 划分数据集的特征 value: 需要返回的特征的值
def splitDataSet(dataSet, axis, value):
    # 为了不修改原始数据集，创建一个新的列表对象
    retDataSet = []
    # 数据集这个列表中的各个元素也是列表，我们要遍历数据集中的每个元素，一旦发现符合要求的值，则将其添加到新创建的列表中
    for featVec in dataSet:
        # 在if语句中，程序将符合特征的数据抽取出来
        if featVec[axis] == value:
            # 抽取掉数据集中目标特征值列
            reducedFeatVec = featVec[ : axis]
            reducedFeatVec.extend(featVec[axis + 1 : ])
            # 将抽取后的数据加入到划分结果列表中
            retDataSet.append(reducedFeatVec)
    return retDataSet

# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
        return bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1

    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    classList = [example[-1] for  example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)

    return myTree
