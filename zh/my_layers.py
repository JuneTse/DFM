# -*- coding: utf-8 -*-
#MyLayers
from keras.layers import LSTM,activations,Layer,Dense,Input,Activation,MaxPooling1D,Flatten,Convolution1D,Merge,InputSpec
from keras.models import Model
from keras.models import K
from keras import  regularizers, constraints,initializers

def to_list(x):
    '''This normalizes a list/tensor into a list.

    If a tensor is passed, we return
    a list of size 1 containing the tensor.
    '''
    if type(x) is list:
        return x
    return [x]

class GatedLayer(Layer):
    '''输入两个2-D的tensor
    '''
    def __init__(self,output_dim,hidden_dim,init='glorot_uniform', activation='linear', weights=None,
                 activity_regularizer=None,input_dim=None, **kwargs):
        '''
        Params:
            output_dim: 输出的维度
        '''
        self.init = initializers.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim=hidden_dim

        self.activity_regularizer = regularizers.get(activity_regularizer)

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(GatedLayer,self).__init__(**kwargs)
        
    def build(self,input_shapes):
        input_shape=input_shapes[0]
        assert len(input_shape)==2
        input_dim=input_shape[1]
        self.input_batch=input_shape[0]
        self.W_s=self.add_weight(shape=(input_dim, self.hidden_dim),
                                      initializer=self.init,
                                      trainable=True,name="{}_W_s".format(self.name))
        
        self.W_e=self.add_weight(shape=(input_dim,self.hidden_dim),initializer=self.init,trainable=True,name="{}_W_e".format(self.name))
        self.b_e=self.add_weight(shape=(self.hidden_dim,),initializer=self.init,trainable=True,name="{}_b_e".format(self.name))
        
        self.W_o=self.add_weight(shape=(self.hidden_dim,self.output_dim),initializer=self.init,trainable=True,name="{}_W_o".format(self.name))
        self.b_o=self.add_weight(shape=(self.output_dim,),initializer=self.init,name="{}_b_o".format(self.name))
        
        
    def call(self,x,mask=None):
        x1=x[0]
        x2=x[1]
        m=K.tanh(K.dot(x1,self.W_s)+K.dot(x2,self.W_e)+self.b_e)
        s=K.sigmoid(K.dot(m,self.W_o)+self.b_o)
        out=self.activation(x1*s+x2*(1-s))
        return out
        
    def get_output_shape_for(self,input_shape):
        return (input_shape[0],input_shape[1],self.output_dim)
    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],self.output_dim)

if __name__=='__main__':
    input_1=Input((20,100))
    input_2=Input((5,100))
    out=AOALayer()([input_1,input_2])
    model=Model([input_1,input_2],out)
    model.compile(optimizer="sgd",loss="mse")
    
    import numpy as np
    data_1=np.random.rand(50,20,100)
    data_2=np.random.rand(50,5,100)
    pre=model.predict([data_1,data_2])
