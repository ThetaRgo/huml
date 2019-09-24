#!/usr/bin/env python
# encoding:utf-8
# @Time   : 2019/9/23
# @Author : 茶葫芦
# @Site   : 
# @File   : main_test.py


from models.knn.Knn import knn_classfier
from sklearn.datasets import load_iris
from pub.model_selection import train_test_split
from pub.processing import meanStdStandardScaler


if __name__ == '__main__':
    mstd=meanStdStandardScaler()
    iris=load_iris()
    x=iris.data
    y=iris.target
    w,v,a,b=train_test_split(x,y)
    mstd.fit(w)
    w=mstd.transform(w)
    a=mstd.transform(a)
    knn=knn_classfier(5)
    knn.fit(w,v)
    y_pre=knn.score(a,b)
    print(y_pre)