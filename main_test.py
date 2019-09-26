#!/usr/bin/env python
# encoding:utf-8
# @Time   : 2019/9/23
# @Author : 茶葫芦
# @Site   : 
# @File   : main_test.py

import numpy as np
from models.knn.Knn import knn_classfier
from sklearn.datasets import load_iris
from pub.model_selection import train_test_split
from pub.processing import meanStdStandardScaler
from models.linearRegression.sample_lnregression import linearRegression

if __name__ == '__main__':
    # KNN
    # mstd=meanStdStandardScaler()
    # iris=load_iris()
    # x=iris.data
    # y=iris.target
    # w,v,a,b=train_test_split(x,y)
    # mstd.fit(w)
    # w=mstd.transform(w)
    # a=mstd.transform(a)
    # knn=knn_classfier(5)
    # knn.fit(w,v)
    # y_pre=knn.score(a,b)
    # print(y_pre)

    #lnrg

    lnrg=linearRegression()
    np.random.seed(100)
    x = np.random.random(1000)
    y = 200 * x + 19 +np.random.normal(size=1000)
    x_n,y_n,x_t,y_t=train_test_split(x,y)
    model=lnrg.fit(x_n,y_n)
    print(model.a_,model.b_)
    print(lnrg.score(y_t,lnrg.predict(x_t)))




