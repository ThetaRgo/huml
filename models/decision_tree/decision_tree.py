#!/usr/bin/env python
# encoding:utf-8
# @Time   : 2019/10/13
# @Author : 茶葫芦
# @Site   : 
# @File   : decision_tree.py

import numpy as np
from collections import Counter
from math import log


class desn_tree(object):

    def split(self, x, y, d, v):
        """
        根据d维度的v值将样本X,Y切分
        :return:
        """
        x_index_more = (x[:, d] >= v)
        x_index_less = (x[:, d] < v)
        return x[x_index_less], x[x_index_more], y[x_index_less], y[x_index_more]

    def entropy(self,y):
        """
        信息熵公式
        :param y:
        :return:
        """
        counter = Counter(y)
        res =0.0
        for keycounts in counter.values():
            p = keycounts / len(y)
            res =res - p * log(p)
        return res

    def gini(self,y):
        """
        基尼系数
        :param y:
        :return:
        """
        counter = Counter(y)
        res =0.0
        for keycounts in counter.values():
            p = keycounts / len(y)
            res = res + p **2
        return 1 -res

    def run(self,x,y):
        """
        对每个维度的每个值进行切分求熵,得到最大的熵的维度和值为切分结果
        :param x:
        :param y:
        :return:
        """
        best_entropy = float('inf')
        best_d, best_v = -1,-1
        #遍历每个维度
        for d in range(x.shape[1]):
            sorted_index = np.argsort(x[:,d])
            for i in range(len(x)):
                if x[sorted_index[i -1],d] !=x[sorted_index[i],d]:
                    v = (x[sorted_index[i -1],d] + x[sorted_index[i],d])/2
                    x_l,x_r,y_l,y_r = self.split(x,y,d,v)
                    entropy_value =self.entropy(y_l) + self.entropy(y_r)
                    # gini_value =self.gini(y_l) + self.gini(y_r)
                    if entropy_value < best_entropy :
                        best_entropy = entropy_value
                        best_d =d
                        best_v =v
        return best_entropy,best_d,best_v


if __name__ == '__main__':
    from sklearn import datasets
    iris = datasets.load_iris()
    x =iris.data[:,2:]
    y=iris.target
    dtree = desn_tree()
    print(dtree.run(x,y))



