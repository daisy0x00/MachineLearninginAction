#coding:utf-8

import bayes
#import re

#listOPosts, listClasses = bayes.loadDataSet()
#myVocabList = bayes.createVocabList(listOPosts)
#print(myVocabList)
#print(bayes.setOfWord2Vec(myVocabList, listOPosts[0]))
#print(bayes.setOfWord2Vec(myVocabList, listOPosts[4]))

#listOPosts, listClasses = bayes.loadDataSet()
#myVocabList = bayes.createVocabList(listOPosts)
#trainMat = []
#for postinDoc in listOPosts:
 #   trainMat.append(bayes.setOfWord2Vec(myVocabList, postinDoc))

#p0V, p1V, pAb = bayes.trainNB0(trainMat, listClasses)

#print(pAb)
#print('*************************************')
#print(p0V)
#print('*************************************')
#print(p1V)

#bayes.testingNB()

# test
#mysent = 'This book is the best book on Python or M.L. I have ever laid eyes upon.'
#print(mysent.split())

#regex = re.compile('\\W*')
#listoftokens = regex.split(mysent)
#print(listoftokens)

bayes.spamTest()