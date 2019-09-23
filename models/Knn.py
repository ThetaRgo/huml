#!/usr/bin/env python
# encoding:utf-8
# @Time   : 2019/9/23
# @Author : 茶葫芦
# @Site   : 
# @File   : Knn.py

import numpy as np
from collections import Counter
from pub.metrics import accuracy

class knn_classfier():
    def __init__(self,k):
        self.k=k


    def fit(self,x,y):
        """knn不需要训练,每次都是全局参与计算"""
        self.train_x=x
        self.train_y = y
        return self

    def predict(self,pre_x):
        # print(pre_x)
        y_predict=[self._predict(x) for x in pre_x]
        return y_predict

    def _predict(self,pre_x):
        distances=[np.sqrt(np.sum((pre_x - x)**2)) for x in self.train_x]
        top_k = np.argsort(distances)[:self.k]
        top_k_y=self.train_y[top_k]
        votes=Counter(top_k_y)
        return votes.most_common(1)[0][0]

    def score(self,x_test,y_test):
        y_predict=self.predict(x_test)
        return accuracy(y_predict,y_test)
