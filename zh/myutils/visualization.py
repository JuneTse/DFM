#coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib import transforms
import matplotlib

matplotlib.use('qt4agg')  
#指定默认字体  
matplotlib.rcParams['font.sans-serif'] = ['SimHei']   
matplotlib.rcParams['font.family']='sans-serif'  
#解决负号'-'显示为方块的问题  
matplotlib.rcParams['axes.unicode_minus'] = False  


def imshow2d(x,cmap=plt.cm.autumn):
    '''
    参数：
        x: 2D
    颜色：
        autumn: 线性增加的 red-orange-yellow
        hot: 
    '''
    plt.imshow(x,cmap=cmap)
    plt.show()
def show_heatmap(data,row_labels=None,column_labels=None):
    fig, ax = plt.subplots(figsize=(data.shape[1]/3,data.shape[0]/3))
    ax.pcolor(data, cmap=plt.cm.Blues)
    
    # put the major ticks at the middle of each cell
    #ax.set_xticks(np.arange(data.shape[1])+0.5, minor=False)
    #ax.set_yticks(np.arange(data.shape[0])+0.5, minor=False)
    
    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    if not row_labels is None:
        ax.set_xticklabels(row_labels, minor=False)
    if not column_labels is None:
        ax.set_yticklabels(column_labels, minor=False)
    plt.show()   

def show_text_attention(text,weights,cmap=cm.Reds,figsize=(1,0.1)):
    '''文本中不同单词显示不同背景颜色
    '''
    plt.figure(figsize=figsize)
    t=plt.gca().transData
    fig=plt.gcf()
    
    x=0
    y=0.5
    
    words=text.split()
    #assert len(words)==len(weights)
    for w,s in zip(words,weights):
        #bbox文本框的属性
        bbox_props=dict(boxstyle="square,pad=0.3",facecolor=cmap(s)[:3],lw=0)
        text=plt.text(x,y," "+w+" ",va='center',ha='left',rotation=0,
                      size=15,bbox=bbox_props,transform=t)
        text.draw(fig.canvas.get_renderer()) #文字画到fig上
        ex=text.get_window_extent() 
        t=transforms.offset_copy(text._transform,x=ex.width,units='dots') #平移
    t=plt.gca().transData
    ax=plt.gca()
    ax.set_axis_off()
    fig=plt.gcf()
    fig.show()
    
if __name__=="__main__":
    a=np.random.rand(2,20)
    
    imshow2d(a)
    b=np.array([[i/20] for i in range(20)])
    imshow2d(b)
    show_heatmap(b)
    
    sent="hello it am a. at the end of the day of the"

    length=len(sent)
    score=[0.1,0.2,0.6,0.4,0.7,0.5,0.8,0.3,0.4,0.2,0.3,0.1]
    show_text_attention(sent,score)