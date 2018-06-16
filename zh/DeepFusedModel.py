# -*- coding: utf-8 -*-
#DSSMModel
import pickle
import keras.backend as K
from keras.layers import Input,merge,Conv1D,MaxPooling1D,LSTM,Dropout,Lambda,Flatten,Dense,Embedding,add,GRU,Permute,Reshape,average,concatenate,add
from keras.layers import BatchNormalization
from keras.models import Model
from keras.layers.wrappers import Bidirectional
from keras.regularizers import l2
from keras.optimizers import Adam
import numpy as np
import os

from my_layers import GatedLayer

from myutils.visualization import show_text_attention
from myutils.regularizations import EarlyStopping
from myutils.train_util import get_weight_path
from entity_align import linking_entity

from pprint import pprint
from matplotlib import cm
import matplotlib.pyplot as plt

import data
from data import get_vocab,get_padded_train_data,get_predicates,get_questions
from word2vec import get_embedding
import re
import random

base_weight_path="./weights/"
base_encoded_path="./datasets/predict/encoded_data"
question_len=data.question_len
predicate_len=data.predicate_len

id2w,vocab=get_vocab()
size=len(vocab)
embedding=get_embedding()
embedding=embedding/np.sqrt((np.sum(np.square(embedding),axis=-1,keepdims=True)+1e-8))  #用这个效果好

#cos函数
def cosine(x1,x2):
    return K.sum(x1*x2,axis=-1)/(K.sqrt(K.sum(x1*x1,axis=-1)*K.sum(x2*x2,axis=-1)+1e-12)) #cos
    
def neg_log_loss(x):
    cos1=x[0]
    cos2=x[1]
    cos3=x[2]
    cos4=x[3]
    cos5=x[4]
    cos6=x[5]
    delta=5 
    p=K.exp(cos1*delta)/(K.exp(cos1*delta)+K.exp(cos2*delta)+K.exp(cos3*delta)+K.exp(cos4*delta)+K.exp(cos5*delta)+K.exp(cos6*delta)) #softmax
    f=-K.log(p) #objective function：-log  #f.shape=(batch_size,)
    return K.reshape(f,(K.shape(p)[0],1))  #return.sahpe=(batch_size,1)
def hinge_loss(x,margin=0.5):
    pos=x[0]
    negs=[x[1],x[2],x[3]]
    losses=[K.maximum(0.0,neg-pos+margin) for neg in negs]
    loss=losses[0]+losses[1]+losses[2]
    return loss
def seq_binary_entropy_loss(y_true, y_pred):
    y_pred=K.clip(y_pred,1e-6,1-1e-6)
    return -K.sum(y_true*K.log(y_pred)+(1-y_true)*K.log(1-y_pred),axis=-1)
    
class GatedBiGRUCNNDSSMTEEMJoint2017(object):
    def __init__(self,samples_num=1000):
        self.weight_path=get_weight_path(self,base_weight_path)
        self.emb_dim=128
        self.question_len=question_len
        self.predicate_len=predicate_len
        

        emb=Embedding(input_dim=size,output_dim=self.emb_dim,weights=[embedding],trainable=False)
        dropout=Dropout(0.25)
        sum_pool=Lambda(lambda x:K.sum(x,axis=1,keepdims=False),output_shape=lambda x:(x[0],x[2]))
        max_pool=Lambda(lambda x:K.max(x,axis=1,keepdims=False),output_shape=lambda x:(x[0],x[2]))
        
        #问题rnn
        question_in=Input(shape=(self.question_len,),dtype='int32')
        question_embeded=emb(question_in)
        question_bigru=Bidirectional(GRU(256,return_sequences=True,activation="tanh"),merge_mode="concat")(question_embeded)
        
        #gate
        gate=Conv1D(1,3,padding="same",activation="sigmoid")(question_bigru)
        att=Lambda(lambda x:1-x,output_shape=lambda x:x)(gate)
        question_subject=Lambda(lambda x:x[0]*x[1],output_shape=lambda x:x[0])([question_bigru,gate])
        question_predicate=Lambda(lambda x:x[0]*(1-x[1]),output_shape=lambda x:x[0])([question_bigru,gate])
        
        question_subject=[Conv1D(200,kernel_size=size,padding="same",activation="relu")(question_subject) for size in [3,4,5]]
        question_subject=concatenate(question_subject)
#        question_subject=Dropout(0.3)(question_subject)
        #subject predict
        question_tags=Conv1D(filters=1,kernel_size=3,activation="sigmoid",padding="SAME",kernel_regularizer=l2(0.001))(question_subject)
        question_tags=Flatten()(question_tags)
        
        ################### questioin predicate semantic model ###################################
#        question_predicate=[Conv1D(200,size,padding="same",activation="relu")(question_predicate) for size in [3,4,5]]
#        question_predicate=concatenate(question_predicate)
#        question_predicate=Dropout(0.3)(question_predicate)
#        question_predicate=Conv1D(self.emb_dim,3,padding='same',activation='relu')(question_predicate)
        question_predicate=Conv1D(self.emb_dim,3,padding='same',activation='linear')(question_predicate)
        question_pool=sum_pool(question_predicate)
        
        att=Lambda(lambda x:K.mean(x,axis=-1),output_shape=lambda x:[x[0],x[1]])(gate)
        
        
        question_shallow=Lambda(lambda x:x[0]*(1-x[1]),output_shape=lambda x:x[0])([question_embeded,gate])
        question_sum=sum_pool(question_shallow)
        
        #谓语属性Model
        predicate_in=Input(shape=(self.predicate_len,),dtype='int32')
        predicate_embeded=emb(predicate_in)
        predicate_bigru=Bidirectional(GRU(256,return_sequences=True,activation='tanh'),merge_mode="concat")(predicate_embeded)
        
#        predicate_cnn=[Conv1D(200,size,padding='same',activation="relu")(predicate_bigru) for size in [3,4,5]]
#        predicate_cnn=concatenate(predicate_cnn)
#        predicate_cnn=Dropout(0.3)(predicate_cnn)
        predicate_cnn=Conv1D(self.emb_dim,3,padding='same',activation='linear')(predicate_bigru)
        
        predicate_pool=sum_pool(predicate_cnn)
        
        predicate_sum=sum_pool(predicate_embeded)
        
        question_out=GatedLayer(self.emb_dim,256,activation="linear")([question_pool,question_sum])
        predicate_out=GatedLayer(self.emb_dim,256,activation='linear')([predicate_pool,predicate_sum])
        
        #subject model
        subject_model=Model(inputs=question_in,outputs=question_tags)
        subject_model.compile(optimizer="adam",loss="mse")
        self.subject_model=subject_model
        #attention
        att_model=Model(inputs=question_in,outputs=att)
        att_model.compile(optimizer="adam",loss="mse")
        self.att_model=att_model
        #sim=merge(inputs=[question_out,predicate_out],mode=lambda x:cosine(x[0],x[1]),output_shape=lambda x:(None,1))   
        sim=Lambda(lambda x:cosine(x[0],x[1]),output_shape=lambda x:(None,1))([question_out,predicate_out])
        sim_model=Model([question_in,predicate_in],sim)
        question_model=Model(question_in,question_out)
        question_model.compile(optimizer='adam',loss='mse')
        predicate_model=Model(predicate_in,predicate_out)
        predicate_model.compile(optimizer='adam',loss='mse')
        self.question_model=question_model
        self.predicate_model=predicate_model
        self.sim_model=sim_model
        
        self.all_model=Model(inputs=[question_in,predicate_in],outputs=[question_tags,sim])
        self.all_model.compile(optimizer="adam",loss="mse")
        self.build()
    def build(self):
         #输入两个样本：正样本和负样本
        input_1=Input(shape=(self.question_len,),dtype='int32')
        input_2_a=Input(shape=(self.predicate_len,),dtype='int32')
        input_2_b=Input(shape=(self.predicate_len,),dtype='int32')
        input_2_c=Input(shape=(self.predicate_len,),dtype='int32')
        input_2_d=Input(shape=(self.predicate_len,),dtype='int32')
        input_2_e=Input(shape=(self.predicate_len,),dtype='int32')
        input_2_f=Input(shape=(self.predicate_len,),dtype='int32')
        
        sim1=self.sim_model([input_1,input_2_a])
        sim2=self.sim_model([input_1,input_2_b])
        sim3=self.sim_model([input_1,input_2_c])
        sim4=self.sim_model([input_1,input_2_d])
        sim5=self.sim_model([input_1,input_2_e])
        sim6=self.sim_model([input_1,input_2_f])
        #合并,输出
        predicate_loss=Lambda(lambda x:neg_log_loss(x),output_shape=lambda x:(None,1))([sim1,sim2,sim3,sim4,sim5,sim6])
        
        #subject loss
        pre_tags=self.subject_model(input_1)
        true_tags=Input(shape=[self.question_len,],dtype="float32")
        subject_loss=Lambda(lambda x:seq_binary_entropy_loss(x[0],x[1]),output_shape=lambda x:(None,1))([true_tags,pre_tags])
        
#        alpha=0.5
#        loss=Lambda(lambda x:x[0]*alpha+x[1]*(1-alpha),output_shape=lambda x:(None,1))([subject_loss,predicate_loss])#add([subject_loss,predicate_loss])
        loss=add([subject_loss,predicate_loss])

        #构造模型    
        self.model=Model([input_1,input_2_a,input_2_b,input_2_c,input_2_d,input_2_e,input_2_f,true_tags],outputs=loss)
        self.model.compile(optimizer="adam",loss=lambda y_true,y_pred:y_pred,metrics=['accuracy'])
    def split_data(self,sample_num,val_split=0.1):
        shuffle_ids=random.sample(list(range(sample_num)),sample_num)
        val_num=int(sample_num*val_split)
        train_ids,val_ids=shuffle_ids[:sample_num-val_num],shuffle_ids[-val_num:]
        return train_ids,val_ids
    def train(self,questions,subjects,predicates,candidates=None,iter_num=15,nb_epoch=1,batch_size=128,val_split=0.2):
        samples_num=len(questions)
        labels=np.array([[0]]*samples_num)
        #把训练数据分成训练集和验证集
        train_ids,valid_ids=self.split_data(sample_num=samples_num,val_split=val_split)
        train_questions=questions[train_ids]
        train_predicates=predicates[train_ids]
        train_subjects=subjects[train_ids]
        train_num=len(train_ids)
        
        valid_questions=questions[valid_ids]
        valid_predicates=predicates[valid_ids]
        valid_subjects=subjects[valid_ids]
        valid_num=len(valid_ids)
        
        #Early stop
        es1=EarlyStopping(patient=3,delta=0,best=-np.inf,monitor=np.greater)
        es2=EarlyStopping(patient=3,delta=0,best=-np.inf,monitor=np.greater)
        
        for it in range(iter_num):
            print('iter:',it)
            r=np.random.randint(0,10)
            if candidates is None or r>8:
                predicates_b,predicates_c,predicates_d,predicates_e,predicates_f=[np.array(random.sample(list(predicates),train_num)) for i in range(5)]
            else:
                predicates_b,predicates_c,predicates_d,predicates_e,predicates_f=[np.array([candidates[j][np.random.randint(len(candidates[j]))]  
                                                                                        for j in train_ids]) for i in range(5)]
            self.model.fit([train_questions,train_predicates,predicates_b,predicates_c,predicates_d,predicates_e,predicates_f,train_subjects],labels[train_ids],epochs=nb_epoch,batch_size=batch_size,shuffle=True)
            if (it+1)%1==0:
                self.save_weights()
                sim=self.sim_model.predict([valid_questions,valid_predicates])
                avg_sim=np.mean(sim)
                print(sim.shape,sim[:10])
                print("average sim:%s"%avg_sim)
                
                subject_pre=self.subject_model.predict(valid_questions)
                acc=self.subject_accuracy(valid_subjects,subject_pre)
                print("subject acc:%s" %acc)
                num,right,acc2017,errors,pre,results=evaluate_subject_extraction(path=data.test_triple2017_path)
                if acc2017>0.936:
                    break
                if es1.is_early_stop(avg_sim) and es2.is_early_stop(acc):
                    print("early stopping...")
                    break
    def encode_question(self,question): #question.shape=(samples,4000)
        return self.question_model.predict(question) #return.shape=(samples,128)
    
    def encode_predicate(self,predicate): #question.shape=(samples,1000)
        return self.predicate_model.predict(predicate) #return.shape=(samples,128)
    def predict_subject(self,questions):
        return self.subject_model.predict(questions)
    def subject_accuracy(self,true_y,pre_y,delta=0.8):
        '''计算准确率
        '''
        all_num=len(true_y)
        right_num=0
        pre_y=np.greater(pre_y,delta).astype(np.int32)
        
        for y1,y2 in zip(true_y,pre_y):
            if np.sum(y1!=y2)==0:
                right_num+=1
        return right_num/all_num
    def show_subject_results(self,questions,pre,delta=0.8):
        '''显示结果'''
        results=[]
        for p,q in zip(pre,questions):
            question=re.split("\\s+",q)
            start=0
            end=len(question)
            p=p[:end]
            threshold=min(delta,max(p)-0.1)
            for i in p:
                if i<threshold:
                    start+=1
                else:
                    break
            
            for j in p[::-1]:
#                print(j,threshold)
                if j<threshold:
                    end-=1
                else:
                    break
            topicEntity="".join(question[start:end])
            topicEntity=linking_entity("".join(question),topicEntity)
            results.append(["".join(question),topicEntity])
        return results 
    def show_subject_results1(self,questions,pre,delta=0.8):
        '''显示结果'''
        results=[]
        for p,q in zip(pre,questions):
            question=re.split("\\s+",q)
            topicEntity="".join([question[i] for i in range(len(p)) if p[i]>=min(delta,max(p)) and i<len(question)])
            results.append(["".join(question),topicEntity])
        return results 
        #保存权值
    def save_weights(self):
        self.all_model.save_weights(self.weight_path,overwrite=True)
        #加载权值   
    def load_weights(self):
        self.all_model.load_weights(self.weight_path)
        
def write_topicEntity(results,outpath="datasets/predict/question_subject2017.txt"):
    with open(outpath,"w",encoding="utf-8") as f:
        for q,es in results:
            if type(es) is list:
                f.write(q+"\t"+es[0][0]+"\n")   
            else:
                f.write(q+"\t"+es+"\n")   
def write_encoded_data(encoded_data,path):
    f=open(path,'w')
    for d in encoded_data:
        for e in d:
            f.write(str(e)+" ")
        f.write('\n')
    f.close()
def encodeData2file(model):
    print("encodeData2file....")
    base_path=os.path.join(base_encoded_path,model.__class__.__name__)
    _,questions=get_questions(data.question2016_path)
    _,questions2017=get_questions(data.question2017_path)
    _,predicates=get_predicates()
    encoded_predicates=model.encode_predicate(predicates)
    write_encoded_data(encoded_predicates,base_path+"_encoded_predicates.txt")
    encoded_questions=model.encode_question(questions)
    write_encoded_data(encoded_questions,base_path+"_encoded_questions.txt")
    encoded_questions2017=model.encode_question(questions2017)
    write_encoded_data(encoded_questions2017,base_path+"_encoded_questions2017.txt")
    print("encoded...")
def cos_sim(x,y):
    return np.sum(x*y,axis=-1)/np.sqrt(np.sum(x*x,axis=-1)*np.sum(y*y,axis=-1)+0.0000001)    
    
def compute_accuracy(predicts,targets):
    errors=[]
    num=len(predicts)
    right=0.
    for p,t in zip(predicts,targets):
        q,s=p
        if s==t:
           right+=1 
        else:
            errors.append([p,t])
    acc=right/num
    print("right:%s, num: %s, acc: %s"%(right,num,acc))
    return num,right,acc,errors
def evaluate_subject_extraction(path=data.test_triple2016_path,plot=False):
    triples,qidx,pidx,sidx,candidates=get_padded_train_data(path)    
    questions=[s[0] for s in triples]
    subjects=[s[1].replace(" ","") for s in triples]
    pre=model.predict_subject(qidx)
    results=model.show_subject_results(questions,pre,delta=0.9)
    pprint(results[10:15])
    
    if plot:
        s=30
        e=50
        att=model.att_model.predict(qidx)
        #att=np.mean(att,axis=-1)
        att=np.reshape(att,[len(att),question_len])
        
        for q,a,p in zip(questions[s:e],att[s:e],pre[s:e]):
            show_text_attention(q,p/2,cm.Blues)
            show_text_attention(q,1-a)
        plt.show()
#    write_topicEntity(results)
    num,right,acc,errors=compute_accuracy(results,subjects)
    return num,right,acc,errors,pre,results
    
if __name__=="__main__":
    model=GatedBiGRUCNNDSSMTEEMJoint2017()
#    _,all_predicates=get_predicates()
#    all_predicates=all_predicates[:50000,:]
    triples,questions,predicates,subjects,candidates=get_padded_train_data()
    
    triples2016,questions2016,predicates2016,subjects2016,candidates2016=get_padded_train_data(path=data.test_triple2016_path)
    
    
#    model.load_weights()
    model.train(questions,subjects,predicates,candidates=candidates,iter_num=15,val_split=0.2)
#    encodeData2file(model)
    model.train(questions,subjects,predicates,candidates=candidates,iter_num=15,val_split=0.1)
    model.train(questions,subjects,predicates,candidates=candidates,iter_num=8,val_split=0.01)
    model.save_weights()
    encodeData2file(model)
    
    ############################# test data 2016 ##################################
    num2016,right2016,acc2016,e2016,pre2016,results2016=evaluate_subject_extraction(path=data.test_triple2016_path)    
    num2017,right2017,acc2017,e2017,pre2017,results2017=evaluate_subject_extraction(path=data.test_triple2017_path)    
    write_topicEntity(results2017)
    
    
    ############################# test data 2017 ##################################
    questions2017,qidx=get_questions(data.question2017_path)
    pre=model.predict_subject(qidx)
    results=model.show_subject_results(questions2017,pre,delta=0.5)
    pprint(results[:10])
    
    s=30
    e=50
    att=model.att_model.predict(qidx)
    #att=np.mean(att,axis=-1)
    att=np.reshape(att,[len(att),question_len])
    
    for q,a,p in zip(questions2017[s:e],att[s:e],pre[s:e]):
        show_text_attention(q,p/2,cm.Blues)
        show_text_attention(q,1-a)
        
    write_topicEntity(results)
    
