#coding:utf-8
import re
import nltk
import itertools
import os
import pickle
from keras.preprocessing import sequence
import numpy as np

train_triple_path="./datasets/train_triples/seg_train_triples.txt"
test_triple2016_path="./datasets/train_triples/seg_test_triples.txt"
test_triple2017_path="./datasets/train_triples/seg_test_triples2017.txt"
kb_path="./datasets/kb/segmented_kbqa.kb"

subject_path="./datasets/kb/all_subjects.txt"
predicate_path="./datasets/kb/all_seg_predicates.txt"

question2017_path="./datasets/questions/all_seg_test_questions2017.txt"
question2016_path="./datasets/questions/all_seg_test_questions.txt"

question_len=20
predicate_len=5

######################## 分词 ####################################
def tokenize(sent):
    words=re.split("\\s+",sent.strip())
    return words

#################################  建立词库  ##################################
def read_questions(path):
    questions=[]
    with open(path,encoding='utf-8') as f:
        for line in f:
            q,s,p,o=line.strip().split("\t")
            words=tokenize(q) #re.split("\\s+",q)
            questions.append(words)
    return questions
def wordReader(path):
    '''读取知识库，每次生成一个词
    '''
    questions=read_questions(train_triple_path)+read_questions(test_triple2016_path)
    for q in questions:
        for w in q:
            yield w
    f=open(path,encoding="utf-8")
    for line in f:
        if line!="":
            line=line.strip()
            words=tokenize(line) #re.split("\\s+",line)
            #去掉无用的符号
            for w in words:
                yield w
    f.close()
def get_vocab(vocab_size=1500000):
    '''统计词频
    '''
    dump_path="./datasets/temp/vocab.pkl"
    if os.path.exists(dump_path):
        return pickle.load(open(dump_path,'rb'))
    else:
        print("building vocab....")
        kb_freqDist=nltk.FreqDist(wordReader(kb_path))
        print(len(kb_freqDist))
    #    questions=read_questions(train_triple_path)+read_questions(test_triple2016_path)
    #    q_freqDist=nltk.FreqDist(itertools.chain(*questions))
        vocab=kb_freqDist.most_common(vocab_size)
        min_count=vocab[-1][1]
        print(min_count)
        vocab=vocab[:vocab_size]
        index2word=['<END>','UNK']+[x[0] for x in vocab]
        word2index=dict([(w,i) for i,w in enumerate(index2word)])
        pickle.dump([index2word,word2index],open(dump_path,"wb"))
        return index2word,word2index
################################### NGram 特征   #######################################
def ngramIsInSubjects(words,subjects,n=2):
    '''以words中每个单词开头的NGram是否在subjects中
    '''
    qlen=len(words)
    v=np.zeros([qlen,])
    for i in range(qlen-n):
        sub=''.join(words[i:i+n])
        if sub in subjects:
            v[i]=1
    return v
def get_ngram_feats(words,subjects):
    feats=[]
    for i in range(1,6):
        feats.append(ngramIsInSubjects(words,subjects,i))
    return np.array(feats)

################################### 训练数据  #################################
def label_subject(question,subject):
    '''标注subject
    '''
    qtemp=question.replace(" ","")
    stemp=subject.replace(" ","")
    assert stemp in qtemp
    qtemp=qtemp.replace(stemp,"X"*len(stemp))
    qtemp=[1 if s=='X' else 0 for s in qtemp]  #字符标注
    
    #标注单词
    labels=[]
    idx=0
    for word in tokenize(question):
        labels.append(qtemp[idx:idx+len(word)])
        idx+=len(word)
#    print(labels)
    labels=[1 if 1 in label else 0 for label in labels]
    return labels
def get_subject_predicates():
    '''subject predicates'''
    dump_path="datasets/temp/subject_predicate.pkl"
    if os.path.exists(dump_path):
        print("loading sp...")
        return pickle.load(open(dump_path,'rb'))
    else:
        sp={}
        with open(kb_path,encoding="utf-8") as f:
            for line in f:
                spo=line.strip().split("\t")
                s=spo[0].strip().replace(" ","")
                s=re.sub("[(（][^\\(\\)（）]+[\\)）]","",s)
                if s not in sp:
                    sp[s]=set()
                sp[s].add(spo[1].strip())
        pickle.dump(sp,open(dump_path,'wb'))
        return sp        
sp=get_subject_predicates()
def get_train_data(path):
    print("loading data...")
    triples=[]
    qidxs=[]
    pidxs=[]
    slabels=[]
    candidates=[]
    _,vocab=get_vocab()
    with open(path,encoding='utf-8') as f:
        for line in f:
            q,s,p,o=line.strip().split('\t')
            #subject 标注
            subject_labels=label_subject(q,s)
            #单词to id
            qidx=[vocab.get(w,1) for w in tokenize(q)]
            pidx=[vocab.get(w,1) for w in tokenize(p)]
#            print(s)
            candidate=[[vocab.get(w,1)  for w in tokenize(c)] for c in sp.get(s.replace(" ","")) if c!=p]
            candidates.append(candidate)
            triples.append([q,s,p,o])
            qidxs.append(qidx)
            pidxs.append(pidx)
            slabels.append(subject_labels)
    return triples,qidxs,pidxs,slabels,candidates
def padding(seq,maxlen,value=0,truncating='pre'):
    '''把数据padding成长度一致的'''
    return sequence.pad_sequences(sequences=seq,maxlen=maxlen,dtype='int32',value=value,truncating=truncating,padding='post')  
def get_padded_train_data(path=train_triple_path):
    triples,qidxs,pidxs,slabels,candidates=get_train_data(path)
    candidates=[padding(c,maxlen=predicate_len,value=0) for c in candidates]
#    triples_test,qidxs_test,pidxs_test,slabels_test=get_train_data(test_triple2016_path)
#    triples=triples+triples_test
#    qidxs=qidxs+qidxs_test
#    pidxs=pidxs+pidxs_test
#    slabels=slabels+slabels_test
    return triples,padding(qidxs,maxlen=question_len,value=0),padding(pidxs,maxlen=predicate_len,value=0),padding(slabels,maxlen=question_len,value=0),candidates
    
################################################ 加载问题    ###################################################
def get_questions(path):
    questions=[]
    qidxs=[]
    _,vocab=get_vocab()
    with open(path,encoding='utf-8') as f:
        for line in f:
            words=tokenize(line.strip())
            qidx=[vocab.get(w,1) for w in words]
            qidxs.append(qidx)
            questions.append(line.strip())
    return questions,padding(qidxs,maxlen=question_len,value=0)
    
def get_predicates(path=predicate_path):
    predicates=[]
    pidxs=[]
    _,vocab=get_vocab()
    with open(path,encoding='utf-8') as f:
        for line in f:
            words=tokenize(line.strip())
            pidx=[vocab.get(w,1) for w in words]
            pidxs.append(pidx)
            predicates.append(line.strip())
    return predicates,padding(pidxs,maxlen=predicate_len,value=0)
    
def read_subjects():
    subjects=set()
    with open(subject_path,encoding='utf-8') as f:
        for line in f:
            subject=line.strip().replace(" ","")
            subjects.add(subject)
    return subjects     
def read_predicates():
    predicates=[]
    with open(predicate_path,encoding='utf-8') as f:
        for line in f:
            predicate=line.strip().replace(" ","")
            predicates.append(predicate)
    return predicates   

##########################  计算每个问题的candidate predicates  ###############################
def get_candidate_predicates(questions,predicates):
    questions=questions/np.sqrt((np.sum(np.square(questions),axis=-1,keepdims=True)+1e-8))
    predicates=predicates/np.sqrt((np.sum(np.square(predicates),axis=-1,keepdims=True)+1e-8))
    sims=np.dot(questions,predicates.T)
    ids=np.argsort(sims,axis=-1)[:,::-1][:,:31]
    return ids,sims

if __name__=="__main__":
    question='安德烈 是 哪个 国家 的 人'
    subject='安德烈'
    labels=label_subject(question,subject)
    print(labels)
    
    id2w,w2id=get_vocab()
    
    triples,qidxs,pidxs,slabels,candidates=get_train_data(train_triple_path)
    
    _,predicates=get_predicates()
    
    
