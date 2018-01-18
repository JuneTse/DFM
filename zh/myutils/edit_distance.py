#coding:utf-8
from pprint import pprint
import numpy as np
'''
计算编辑距离：使用动态规划
'''

def calEditDist(words1,words2):
    '''
    计算编辑距离,算法如下：
    1. 初始化一个二维数组d[m][n], 其中d[i][j]表示words1[:i]与words2[:j]的编辑距离
    2. d[0][j]=j , j in range(0,n)
       d[i][0]=i , i in range(0,m)
    3.for i in range(1,m):
        for j in range(1,n):
            d[i][j]=min{d[i-1][j]+1,d[i][j-1]+1, d[i][j]+0?word1[i]==words2[j]:1}
    '''
    m=len(words1)
    n=len(words2)
    d=[[-1 for i in range(n)] for m in range(m)]
    for i in range(m):
        d[i][0]=i
    for j in range(n):
        d[0][j]=j
    #pprint(np.array(d))
    for i in range(1,m):
        for j in range(1,n):
            minValue=d[i-1][j-1]
            if not words1[i-1]==words2[j-1]:
                minValue+=1
            if d[i-1][j]+1<minValue:
                minValue=d[i-1][j]+1
            if d[i][j-1]+1<minValue:
                minValue=d[i][j-1]+1
            d[i][j]=minValue
    #pprint(np.array(d))            
    return d[m-1][n-1]
    
def calWeightedEditDist(words1,words2,weights1=None,weights2=None):
    '''计算加权的编辑距离'''

    m=len(words1)
    n=len(words2)
    if weights1 is None:
        weights1=[1]*m
    if weights2 is None:
        weights2=[1]*n
    d=[[-1 for i in range(n)] for m in range(m)]
    for i in range(m):
        d[i][0]=i
    for j in range(n):
        d[0][j]=j
    #pprint(np.array(d))
    for i in range(1,m):
        for j in range(1,n):
            minValue=d[i-1][j-1]
            if not words1[i-1]==words2[j-1]:
                minValue+=min([weights1[i],weights2[j]])
            if d[i-1][j]+1<minValue:
                minValue=d[i-1][j]+weights1[i]
            if d[i][j-1]+1<minValue:
                minValue=d[i][j-1]+weights2[j]
            d[i][j]=minValue
    #pprint(np.array(d))            
    return d[m-1][n-1]
    
if __name__=="__main__":
    words1="戴维斯"
    words2="哈维斯"
    d=calEditDist(words1,words2)
    print(d)
    
    weights1=[1,1,0.9]
    d=calWeightedEditDist(words1,words2,weights1)
    print(d)