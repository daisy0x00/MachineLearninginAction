#coding:utf-8

import tree

# 计算数据集的香农熵
#myDat, labels = tree.createDataSet()
#print(myDat)
#print(tree.calcShannonEnt(myDat))
#myDat[0][-1] = 'maybe'
#print(myDat)
#print(tree.calcShannonEnt(myDat))

print("*****************************************************************")
# 在前面的简单样本数据上测试函数splitDataSet()
#print(tree.splitDataSet(myDat, 0, 1))
#print(tree.splitDataSet(myDat, 0, 0))

print("*****************************************************************")
#myDat, labels = tree.createDataSet()
#print(myDat)
#print(tree.chooseBestFeatureToSplit(myDat))

print("*****************************************************************")
# 变量myTree包含了很多代表树结构信息的嵌套字典。
# 从左边开始第一个关键字no surfacing是第一个划分数据集的特征名称，该关键字的值也是另一个数据字典。
# 第二个关键字是no surfacing特征划分的数据集，这些关键字的值是no surfacing节点的子节点
# 这些值可能是类标签，也可能是另一个数据字典。如果值是类标签，则该子节点是叶子节点；
# 如果值是另一个数据字典，则子节点是一个判断节点，这种格式结构不断重复就构成了整棵树，本节的例子中，这棵树包含了3个叶子节点以及2个判断节点
myDat, labels = tree.createDataSet()
myTree = tree.createTree(myDat, labels)
print(myTree)

