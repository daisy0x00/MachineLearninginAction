#coding:utf-8

from numpy import *

# 词表到向量的转换函数
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'uying', 'worthless', 'dog', 'food', 'stupid']]

    classVec = [0, 1, 0, 1, 0, 1]  # 1代表侮辱性文字， 0代表正常言论

    return postingList, classVec

def createVocabList(dataSet):
    # 创建一个空集
    vocabSet = set([])


    for document in dataSet:
        # 创建两个集合的并集
        # 操作符|用于求两个集合的并集，这也是一个按位或（OR）操作符
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

# 输入参数为词汇表及某个文档
def setOfWord2Vec(vocabList, inputSet):
    # 创建一个和词汇表等长的向量，并将其元素都设置为0
    returnVec = [0] * len(vocabList)
    # 遍历文档中的所有单词，如果出现了词汇表中的单词，则将输出的文档向量中的对应值设为1
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word: %s is not in my Vocabulary!' % word)
    # 输出文档向量，向量的每一元素为1或者0，分别表示词汇表中的单词在输入文档中是否出现
    return returnVec

# 朴素贝叶斯分类器训练函数
# trainMatrix:文档矩阵
# trainCategory：由每篇文档类别标签所构成的向量
def trainNB0(trainMatrix, trainCategory):
    # 计算p(wi|c1)和p(wi|c0),需要初始化程序中的分子变量和分母变量
    # 文档矩阵的长度，即测试实例的个数
    numTrainDocs = len(trainMatrix)
    # 一个测试实例中的词汇量
    numWords = len(trainMatrix[0])
    # 标签向量中1的个数之和即是侮辱性实例的个数，numTrainDocs表示测试实例的个数，相除可以算出本次测试中侮辱性留言的概率p1
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 用zeros创建初值为0.0的一维数组，数组元素类型默认为浮点型
    # p0Num用来存储某个词在类别0中出现的次数
    # p1Num用来存储某个词在类别1中出现的次数
    #p0Num = zeros(numWords)
    #p1Num = zeros(numWords)
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    # 初始化概率
    # p0Denom用来存储类别0中出现的所有词汇的总数
    # p1Denom用来存储类别1中出现的所有词汇的总数
    p0Denom = 2.0
    p1Denom = 2.0
    # 遍历文档矩阵trainMatrix的每一行元素
    for i in range(numTrainDocs):
        # 如果该行元为类别1
        if trainCategory[i] == 1:
            # 向量相加
            #
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 对每个元素除以该类别中的总词数
    p1Vect = p1Num / p1Denom
    p0Vect = p0Num / p0Denom

    # 函数返回两个向量和一个概率
    return p0Vect, p1Vect, pAbusive

# 朴素贝叶斯分类函数
def classifyNB(vec2Classify, p0vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWord2Vec(myVocabList, postinDoc))

    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWord2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWord2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

# 文件解析及完整的垃圾邮件测试函数
# 函数textParse接受一个大字符串并将其解析为字符串列表。
# 该函数去掉少于两个字符的字符串，并将所有字符串转换为小写
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

# 函数spamTest()对贝叶斯垃圾邮件分类器进行自动化处理。
# 导入文件夹spam与ham下的文本文件，并将它们解析为词列表。
def spamTest():
    docList = []
    classList = []
    fullText = []
    # 导入并解析文本文件
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' %i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    # trainingSet是一个整数列表，其中的值从0到49
    trainingSet = list(range(50))
    testSet = []
    # 选取10封电子邮件，随机构建训练集
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    # 对测试集分类
    for docIndex in trainingSet:
        trainMat.append(setOfWord2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWord2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ', float(errorCount) / len(testSet))
