#!/usr/bin/env python
# encoding:utf-8
# @Time   : 2019/9/23
# @Author : 茶葫芦
# @Site   : 
# @File   : main_test.py


from models.Knn import knn_classfier
from sklearn.datasets import load_iris
from pub.model_selection import train_test_split


if __name__ == '__main__':
    iris=load_iris()
    x=iris.data
    y=iris.target
    w,v,a,b=train_test_split(x,y)
    knn=knn_classfier(5)
    knn.fit(w,v)
    y_pre=knn.score(a,b)
    print(y_pre)