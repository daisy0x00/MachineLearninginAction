#coding:utf-8

import random
from numpy import *
import os
# SMO算法中的辅助函数

# 辅助一：loadDataSet()函数，用于打开文件并对其进行逐行解析，从而得到每行的列标签和整个数据矩阵
def loadDataSet(fileName):
    dataMat = []
    labelMat = []

    fr = open(fileName)

    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))

    return dataMat, labelMat

# 辅助二：selectJrand()函数，该函数有两个参数，i是第一个alpha的下标，m是所有alpha的数目
# 该函数的作用是，只要函数值不等于输入值i，函数就会进行随机选择
def selectJrand(i, m):
    j = i
    while(j == i):
        j = int(random.uniform(0, m))
    return j

# 辅助三：clipAlpha()函数，该函数用于调整大于H或者小于L的alpha值。使求得的alpha值满足0 < alpha < C这个约束条件
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj

# 简化版SMO算法
# 参数介绍：dataMatIn（数据集）， classLabels（类别标签）， C（常数C），toler（容错率）， maxIter（退出前最大的循环次数）
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    # 我们在构建函数时采用了通用的接口，这样就可以对算法和数据源进行组合或者配对处理
    # 初始化数据
    # 将数据集转换成NumPy矩阵
    dataMatrix = mat(dataMatIn)
    # 将类别标签的列表转化为NumPy矩阵并进行转置
    # 此时的类型标签不是列表而是一个列向量，类型标签向量的每一行元素都和数据矩阵中的行一一对应
    labelMat = mat(classLabels).transpose()
    # 初始化b为0
    b = 0
    # 通过矩阵dataMatIn的shape属性得到常数m和n
    m,n = shape(dataMatrix)
    # 构建一个alphas列矩阵，矩阵中的元素都初始化为0
    alphas = mat(zeros((m, 1)))
    # 建立一个iter变量，该变量存储的是在没有任何alpha改变的情况下遍历数据集的次数。
    # 如果该变量达到输入值maxIter时，函数结束运行并退出
    iter = 0

    # 简化版SMO算法将寻找alpha对这一步简化，直接按照遍历数据集的方法来找alphai
    while(iter < maxIter):
        # alphaPairsChanged用来记录alpha是否已经进行优化
        # 每次循环中，将alphaPairsChanged先设为0，然后再对整个集合顺序遍历
        alphaPairsChanged = 0
        for i in range(m):
            # 第i个样本的预测结果
            fXi = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            # 计算第i个样本的预测结果与实际结果之差
            Ei = fXi - float(labelMat[i])
            #
            if(labelMat[i] * Ei < -toler) and (alphas[i] < C) or (labelMat[i] * Ei > toler) and (alphas[i] >0):
                # 随机选择第二个alpha值
                j = selectJrand(i, m)
                # 第j个样本的预测结果
                fXj = float(multiply(alphas, labelMat).T *(dataMatrix * dataMatrix[j, :].T)) + b
                # 计算第j个样本的预测结果与实际结果之差
                Ej = fXj - float(labelMat[j])
                # 用copy()明确告知python要为alphaIold和alphaJold分配新内存，以便在新旧值做比较时看到新旧值的变化
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                # 这一部分主要根据第i个样本的实际类别和第j个样本的实际类别来确定alpha的边界值L和H，以使得alphai和alphaj满足0 < alpha < C
                if(labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L == H")
                    continue

                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T - \
                      dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    print("eta >= 0")
                    continue

                # alphaj在沿着约束方向未剪辑时的值，即未考虑约束条件0 <= alpha <= C 时求出的alphaj的最优解
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                # alphaj考虑约束条件0 <= alpha <= C 后求出的alphaj的最优解
                alphas[j] = clipAlpha(alphas[j], H, L)

                if(abs(alphas[j] - alphaJold) < 0.00001):
                    print('j not moving enough')
                    continue
                # alphai的最优解
                alphas[i] += labelMat[i] * labelMat[j] * (alphaJold - alphas[j])

                # 设置常数项b
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[j, :] * dataMatrix[j, :].T - \
                    labelMat[j] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T
                if(0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif(0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0

                alphaPairsChanged += 1
                print("iter: %d i: %d, pairs changed %d" %(iter, i, alphaPairsChanged))

        # 在for循环之外，需要检查alpha值是否做了更新，如果有更新则将iter设为0后继续运行程序。
        # 只有在所有数据集上遍历maxIter次，且不再发生任何alpha修改之后，程序才会停止并退出while循环
        if(alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" %iter)
    return b, alphas


# 完整版Platt SMO的支持函数
# 在讲述改进后的代码之前，我们必须对上节的代码进行清理，下面的程序清单中包含一个用于清理代码的数据结构和3个用于对E进行缓存的辅助函数

# 用于清理代码的数据结构
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        # eCache的第一列给出的是eCache是否有效的标志位，第二列给出的是实际的E值
        self.eCache = mat(zeros((self.m, 2)))


# calcEk()函数对给定的alpha值能计算E值并返回
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.X * oS.X[k, :].T) + oS.b
    Ek = fXk - float(oS.labelMat[k])

    return Ek

# selectJ()用于选择第二个alpha值或者说内循环的alpha值
# 这里的目标是选择合适的第二个alpha值以保证在每次优化中采用最大步长
def selectJ(i, oS, Ei):
    # 初始化要返回的参数
    # 初始化选择的第二个alpha对应的下标
    maxK = -1
    # 初始化选择的第二个alpha与第一个alpha值对应的步长
    maxDeltaE = 0
    # 初始化Ej的值
    Ej = 0
    # 将Ei在缓存中设置成为有效的，即Ei是已经计算好的值
    oS.eCache[i] = [1, Ei]
    # 该代码在eCache中构建出一个非零表
    # nonzero()语句返回的是非零E值所对应的alpha值，而不是E值本身
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
    if(len(validEcacheList)) > 1:
        # 程序会在所有的值上进行循环并选择其中使得改变最大的那个值
        for k in validEcacheList:
            # 如果遍历到的值和第一个变量的值相等，直接跳出本次循环
            if k == i:
                continue
            # 计算Ek
            Ek = calcEk(oS, k)
            # 选择具有最大步长的j
            deltaE = abs(Ei - Ek)
            if(deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    # 如果这是第一次循环的话，那么就随机选择一个alpha值
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

# updataEk()函数用于计算误差值，并存入缓存中，在对alpha值进行优化后会用到该值
def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]



# 完整Platt SMO算法中的优化例程
# 该代码几乎和smoSimple()函数一模一样，主要做了如下几方面修改。
# 1.使用了自己的数据结构，该结构在参数oS中传递。
# 2.使用selectJ()而不是selectJrand()来学则第二个alpha值
# 3.在alpha值改变时更新Ecache
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or \
            ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        # 第二个alpha值选择中的启发式方法
        j, Ej = selectJ(i, oS, Ei)
        #
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()

        if(oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.alphas[j] - oS.alphas[i] + oS.C)
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H :
            print("L == H")
            return 0

        # eta = - ||xi - xj||^2(我目前还不知道这代表什么，但是求沿着约束方向未剪辑的alphaj的公式里确实有这么一项)
        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T
        if eta >= 0:
            print("eta >= 0")
            return 0
        # 沿着约束方向未剪辑的alphaj，即未考虑约束条件0 < alphas < C时的alphaj值
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        # 加上约束条件0 < alphas < C后的alphaj值
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        # 在求出alphaj值时更新Ej
        updateEk(oS, j)
        # 查看alphaj新旧值的变化
        if(abs(oS.alphas[j] - alphaJold ) < 0.00001):
            print("j not moving enough")
            return 0

        # alphai优化后的值
        oS.alphas[i] += oS.labelMat[i] * oS.labelMat[j] * (alphaJold - oS.alphas[j])
        # 在求出优化后的alphai更新Ei
        updateEk(oS, i)
        # 求更新后的b
        b1 = oS.b - Ei - oS.labelMat[i] * oS.X[i, :] * oS.X[i, :].T * (oS.alphas[i] - alphaIold) - \
            oS.labelMat[j] * oS.X[i, :] * oS.X[j, :].T * (oS.alphas[j] - alphaJold)
        b2 = oS.b - Ej - oS.labelMat[j] * oS.X[j, :] * oS.X[j, :].T *(oS.alphas[j] - alphaJold) - \
            oS.labelMat[i] * oS.X[i, :] * oS.X[j, :].T * (oS.alphas[i] - alphaIold)

        if(0 < oS.alphas[i]) and(oS.alphas[i] < oS.C):
            oS.b = b1
        elif(0 < oS.alphas[j]) and(oS.alphas[j] < oS.C):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


# 完整版Platt SMO的外循环代码，即选择第一个alpha值的外循环
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup = ('lin', 0)):
    # 构建一个数据结构来容纳所有的数据
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while(iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            # 遍历所有值
            for i in range(oS.m):
                # innerL()启发式方法找alpha2，并在可能时对其进行优化处理，若有任意一对alpha值发生改变则返回1
                alphaPairsChanged += innerL(i, oS)
                print("fullSet, iter: %d i: %d, pairs changed %d" %(iter, i, alphaPairsChanged))
            iter += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter: %d i: %d, pairs changed %d" %(iter, i, alphaPairsChanged))
            iter += 1

        if entireSet:
            entireSet = False
        elif(alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" %iter)
    return oS.b, oS.alphas



# w的计算
def calcWs(alphas,dataArr, classLabels):
    X = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w


