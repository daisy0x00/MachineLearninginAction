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
            # 扩展字典并将当前键值加入字典
            labelCounts[currentLabel] = 0
        # 当前类别出现的次数加一
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
    # 数据集这个列表中的各个元素也是列表，我们要遍历数据集中的每个元素
    # 一旦发现符合要求的值，则将其添加到新创建的列表中
    # 即一旦找到某个特征值等于传进来的参数特征值，就把这个实例中的那个特征值抽取掉，如何把抽取后的数据加入到划分结果列表中
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
# 该函数实现选取特征，划分数据集，计算得出最好的划分数据集的特征
# 在函数中调用的数据需要满足一定的要求：
# 第一个要求是，数据必须是一种由列表元素组成的列表，而且所有的列表元素都要具有相同的数据长度。
# 第二个要求是，数据的最后一列或者每个实例的最后一个元素是当前实例的类别标签。
# 数据集一旦满足上述要求，我们就可以在函数的第一行判定当前数据集包含多少特征属性。
# 我们无需限定list中的数据类型，他们既可以是数字也可以是字符串，并不影响实际计算、
def chooseBestFeatureToSplit(dataSet):
    # 每个元素含有的特征值个数，由于最后一个元素是当前实例的类别标签，因此用列表长度减去1
    numFeatures = len(dataSet[0]) - 1
    # 计算了整个数据集的原始香农熵，我们保存最初的无序度量值，用于与划分完之后的数据集计算的熵值进行比较
    baseEntropy = calcShannonEnt(dataSet)
    # 初始化最大信息增益值为零
    bestInfoGain = 0.0
    # 初始化可选择的最好的特征值为-1
    bestFeature = -1
    # 遍历数据集中的所有特征
    for i in range(numFeatures):
        # 使用列表推导来创建新的列表，将数据集中所有第i个特征值或者所有可能存在的值写入这个新list中
        # 就是抽取每个实例的第i个特征值，用列表推导把这个元素放到新的list中
        featList = [example[i] for example in dataSet]
        # 使用Python语言原生的集合（set）数据类型。集合数据类型与列表类型相似，不同之处在于集合类型中的每个值互不相同
        # 从列表中创建集合是Python语言得到列表中唯一元素值的最快方法
        uniqueVals = set(featList)
        #
        newEntropy = 0.0
        #遍历当前特征中的所有唯一属性值，对每个唯一属性值划分一次数据集
        # 然后计算数据集的新熵值，并对所有唯一特征值得到的熵求和
        # 信息增益是熵的减少或者是数据无序度的减少，最后比较所有特征中的信息增益，返回最好特征划分的索引值
        for value in uniqueVals:
            #对每个唯一属性值划分一次数据集
            # splitDataSet()函数的返回值是抽取数据集中目标特征列值之后的列表
            subDataSet = splitDataSet(dataSet, i, value)
            # ?
            prob = len(subDataSet) / float(len(dataSet))
            # 新熵值？prob不理解
            newEntropy += prob * calcShannonEnt(subDataSet)

        # 本次信息增益
        infoGain = baseEntropy - newEntropy
        # 保持bestInfoGain里面始终放的是最大的信息增益
        # 同时更新最好的划分特征
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
        # 返回最好的划分特征
        return bestFeature


# 决策树的工作原理如下：得到原始数据集，然后基于最好的属性值划分数据集，
# 由于特征值可能多于两个，因此可能存在大于两个分支的数据集划分。
# 第一次划分之后，数据将被向下传递到树分支的下一个节点，在这个节点上，
# 我们可以再次划分数据，因此我们可以采用递归的原则处理数据集。
# 递归结束的条件是：程序遍历完所有划分数据集的属性，或者每个分支下的所有实例都具有相同的分类。
# 如果所有实例具有相同的分类，则得到一个叶子节点或者终止块。
# 任何到达叶子节点的数据必然属于叶子节点的分类

# 采用多数表决的方法决定该叶子节点的分类
# 该函数使用分类名称的列表，然后创建键值为classList中唯一值的数据字典，
# 字典对象存储了classList中每个类标签出现的频率，最后利用operator操作键值排序字典
# 并返回出现次数最多的分类名称。
def majorityCnt(classList):
    #创建键值为classList中唯一值的数据字典
    classCount = {}
    # 创建字典的过程
    # 字典对象存储了classList中每个类标签出现的频率
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1

    # 利用operator操作键值排序字典
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
    # 返回出现次数最多的分类名称
    return sortedClassCount[0][0]

# 创建树的函数代码
# dataSet：数据集， labels: 标签列表
def createTree(dataSet, labels):
    # classList列表中包含了数据集的所有类标签
    classList = [example[-1] for  example in dataSet]
    # 递归函数的第一个停止条件是所有的类标签完全相同，则直接返回该类标签
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 递归函数的第二个停止条件是使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组
    # 由于第二个条件无法简单地返回唯一的类标签，这里使用前面介绍的majorityCnt函数挑选出现次数最多的类别作为返回值
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    # 下面开始创建树，这里使用Python语言的字典类型存储树的信息
    # 变量bestFeat存储当前数据集选取的最好特征
    bestFeat = chooseBestFeatureToSplit(dataSet)
    #
    bestFeatLabel = labels[bestFeat]
    # 字典变量mytree存储了树的所有信息
    myTree = {bestFeatLabel:{}}
    #
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)

    # 遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用函数createTree(),
    # 得到的返回值将被插入到字典变量myTree中，因此函数终止执行时，字典中将会嵌套很多代表叶子节点信息的字典数据
    # 在解释这个嵌套数据之前，我们先看一下循环的第一行subLabels = labels[:]，这行代码复制了类标签，
    # 并将其存储在新列表变量subLabels中。之所以这样做，是因为在Python语言中函数参数是列表类型时，
    # 参数是按照引用方式传递的。为了保证每次调用函数createTree()时不改变原始列表的内容，使用新变量subLabels代替原始列表。
    for value in uniqueVals:
        # 复制类标签，并将其存储在新列表变量subLabels中
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)

    return myTree
