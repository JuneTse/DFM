#coding:utf-8
import sys
import numpy as np
import data
from data import read_subjects,read_predicates
from myutils.edit_distance import calEditDist
import os
import pickle
from pprint import pprint
import re

stopwords=[]
with open("datasets/stopwords.txt",encoding='utf-8') as f:
    for line in f:
        stopwords.append(line.strip())

'''
实体对齐：
问题中抽取的主题实体与知识库中的subject对齐，解决抽取错误和主题实体与subject不一致的问题
'''

def build_index(subjects):
    '''建立倒排索引
    '''
    dump_path=os.path.join("datasets/temp/vocab_index.pkl")
    if os.path.exists(dump_path):
        return pickle.load(open(dump_path,"rb"))
    else:
        subjects=list(subjects)
        char_index={}
        for i in range(len(subjects)):
            chars=list(subjects[i])
            for c in set(chars):
                if not c in char_index:
                    char_index[c]=[i]
                else:
                    char_index[c].append(i)
        pickle.dump(char_index,open(dump_path,"wb"))
        return char_index
    
def stat_overlap(entity,subjects,char_index):
    '''统计重叠字的个数'''
    freq=np.zeros([len(subjects),])
    for c in list(entity):
        freq[char_index.get(c)]+=1
    am=freq.argmax()
    mx=freq[am]
    idx=np.where(freq==mx)
    return np.array(list(subjects))[idx],idx
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
                minValue=d[i][j-1]+weights2[j]*1.0/n
            d[i][j]=minValue
    #pprint(np.array(d))            
    return d[m-1][n-1]    
def getNGrams(question):
    '''所有可能的n_grams
    '''
    #words=re.split("\\s+",question)
    words=list(question.replace(" ",""))
    n_grams=[]
    l=len(words)
    for n in list(range(1,l+1))[::-1]:
        for i in range(l-n+1):
            ngram=words[i:i+n]
            s="".join(ngram)
#            if s in subject_stopwords:
#                continue
            n_grams.append(s)
    return set(n_grams)
    
def ngram_filter(ngrams):
    '''过滤掉子串
    '''    
    results=[]
    for ng in ngrams:
        flag=0
        word="".join(ng)
        if word.endswith("？") or word.endswith("?") or word.find("《")!=-1 or word.find("》")!=-1:
            continue
        for ng1 in ngrams:
            word1="".join(ng1)
            #如果word是word1的子串，则过滤掉
            if word!=word1 and word in word1: #and re.sub("[a-zA-Z0-9]","",word).strip()=="" :
                flag=1
                break
        if flag==0:
            results.append(ng)
    return results   
predicates=read_predicates()
predicates=set(predicates[:2000])
subjects=read_subjects()
def linking_entity(question,entity,subjects=subjects,weights=None):
    '''
    1. 取所有n-gram与subjects的交集
    2. 计算交集与抽取的entity的编辑距离
    '''
    if entity in subjects:
        return entity
    else:
        ngrams=getNGrams(question)
        ng=list(ngrams&subjects)
        ng=[g for g in ng if g not in predicates and g not in stopwords]
        editDists=[]
        candidates=[]
        ng=ngram_filter(ng)
        for g in ng:
            d=calWeightedEditDist(entity,g,weights1=weights)
            editDists.append(d)
            candidates.append(g)
        res=sorted(zip(candidates,editDists),key=lambda x:x[1])
#        print(res)
        if len(res)>0 and res[0][1]<len(entity)*0.8:
            return res[0][0]
        else:
            return entity
            
if __name__=="__main__":
    subjects=read_subjects()
    char_index=build_index(subjects)
    word,idx=stat_overlap("2012款凯美瑞200g经典",subjects,char_index)
    pprint(word)
    
    question="科学的解释猫的类别"
    entity="科学的解释猫"
    e=linking_entity(question,entity,subjects)
    print(e)
