#coding:utf-8

import svmMLiA
from numpy import *

#dataArr, labelArr = svmMLiA.loadDataSet('testSet.txt')
#print(labelArr)
#b, alphas = svmMLiA.smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
#print(b)
#print(alphas[alphas > 0])

dataArr, labelArr = svmMLiA.loadDataSet('testSet.txt')
b, alphas = svmMLiA.smoP(dataArr, labelArr, 0.6, 0.001, 40)
#print(b)
#print(alphas)
print(alphas[alphas > 0])

ws = svmMLiA.calcWs(alphas, dataArr, labelArr)
print(ws)

dataMat = mat(dataArr)
print(dataMat[0] * mat(ws) + b)
print(labelArr[0])