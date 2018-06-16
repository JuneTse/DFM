#coding:utf-8
import re

def tokenize(sent):
    words=re.split("\\s+",sent.strip())
    return words

#################################  建立词库  ##################################
def read_questions(path):
    questions=[]
    with open(path,encoding='utf-8') as f:
        for line in f:
            words=tokenize(line.strip()) #re.split("\\s+",q)
            questions.append(words)
    return questions
    
quetions=read_questions("all_seg_questions.txt")
test_quetions=read_questions("all_seg_test_questions.txt")
test_quetions2017=read_questions("all_seg_test_questions2017.txt")

all_questions=quetions+test_quetions+test_quetions2017

def stat_words():
    ws={}
    for words in all_questions:
        for w in words:
            if w not in ws:
                ws[w]=1
            else:
                ws[w]+=1
    words=sorted(ws.items(),key=lambda x:x[1],reverse=True)
    return words
    
words=stat_words()
stopwords=[w for w,i in words[:200]]
with open("../stopwords.txt","w",encoding="utf-8") as f:
    for w in stopwords:
        f.write(w+"\n")
    
