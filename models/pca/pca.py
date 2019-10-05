#!/usr/bin/env python
# encoding:utf-8
# @Time   : 2019/10/4
# @Author : 茶葫芦
# @Site   : 
# @File   : pca.py

import numpy as np
import matplotlib.pyplot as plt


class pca():
    def __init__(self,initial_w,n_compents,eta =0.1,epsilon=1e-10,n_iters=1e8):
        self.initial_w=initial_w
        self.n_compents=n_compents
        self.eta=eta
        self.epsilon=epsilon
        self.n_iters=n_iters

    def demean(self,X):
        # 对数据进行平均值为0化
        return X -np.mean(X,axis=0)

    def first_compent(self,X):

        def f(w,X):
            # 目标函数(损失函数,X为demean处理后的样本)
            return np.sum((X.dot(w))**2)/len(X)
        def df(w,X):
            # 目标函数导数
            return X.T.dot(X.dot(w))* 2./len(X)

        def df_debug(w,X,epsilon=0.00001):
            # 根据梯度定义而产生的一个通用的梯度求法(与函数形态无关,可用于梯度公式验证)
            res = np.empty(len(w))
            for i in range(len(w)):
                w_1 =w.copy()
                w_1[i]+=epsilon
                w_2=w.copy()
                w_2[i] -=epsilon
                res[i] =(f(w_1,X) -f(w_2,X)) / (2 * epsilon)
            return res

        def unit_direction(w):
            #将一个向量转化成单位向量
            return w /np.linalg.norm(w)

        # 梯度上升法求最大值
        cur_iter = 0
        self.initial_w = unit_direction(self.initial_w)
        while cur_iter < self.n_iters:
            last_y = f(self.initial_w,x)
            gradient= df(self.initial_w,x)
            self.initial_w = self.initial_w + gradient * self.eta
            self.initial_w = unit_direction(self.initial_w)
            if last_y - f(self.initial_w ,x) <= self.epsilon:
                break
            cur_iter +=1
        return self.initial_w

    def first_n_compents(self,X):
        res =[]
        for i in range(self.n_compents):
            w = self.first_compent(X)
            res.append(w)
            X - X.dot(w).reshape(-1,1)*w

        return res



if __name__ == '__main__':

    np.random.seed(1000)
    x = np.empty((100,2))
    x[:,0]= np.random.uniform(0,100,size=100)
    x[:,1]=0.75 * x[:,0] +3.
    print(x)

    w =np.random.random(2)
    p=pca(initial_w=w,n_compents=2)
    res =p.first_n_compents(x)

    print(res)

