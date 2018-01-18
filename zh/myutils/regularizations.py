#coding:utf-8
import numpy as np

class EarlyStopping(object):
    def __init__(self,patient=2,delta=0.0,best=np.inf,monitor=np.less):
        '''
        参数：
            patient: 满足结束条件的次数。
            best: 最好的结果,默认为np.inf, 越小越好
            monitor: 判断器，用于判断是否满足结束条件
        '''
        self.patient=patient
        self.delta=delta
        self.best=best
        self.monitor=monitor
        self.wait=0
        
    def is_early_stop(self,cur):
        '''
        参数：
            cur:当前的值,与best比较
        '''
        if self.monitor(cur+self.delta,self.best):
            self.wait=0
            if self.monitor(cur,self.best):
                self.best=cur
        else:
            if self.wait>=self.patient:
                return True
            self.wait+=1
            
        return False
        
        