#coding:utf-8

def compute_accuracy(predicts,targets):
    num=len(predicts)
    right=0.
    for p,t in zip(predicts,targets):
        if p==t:
           right+=1 
    acc=right/num
    print("right:%s, num: %s, acc: %s"%(right,num,acc))
    return num,right,acc