#!/usr/bin/env python
# encoding:utf-8
# @Time   : 2019/10/13
# @Author : 茶葫芦
# @Site   : 
# @File   : svm.py

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris,make_moons
from sklearn.model_selection import train_test_split


def linear_svc_pipeline(c=1):
    return Pipeline([
            ('std_scaler', StandardScaler()),
            ('linearsvc', LinearSVC(C=c)),

        ])

# 非线性
def svc_pipeline():
    return Pipeline([
            ('std_scaler', StandardScaler()),
            ('svc', SVC(kernel='rbf')),

        ])


if __name__ == '__main__':
    iris =load_iris()
    y =iris.target
    x =iris.data
    x_test,x_train,y_test,y_train = train_test_split(x,y)
    p =linear_svc_pipeline()
    p.fit(x_train,y_train)
    print(p.score(x_test,y_test))


    x,y =make_moons()
    x_test, x_train, y_test, y_train = train_test_split(x, y)
    svc =svc_pipeline()
    svc.fit(x_train,y_train)
    print(svc.score(x_test, y_test))