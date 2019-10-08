#!/usr/bin/env python
# encoding:utf-8
# @Time   : 2019/10/6
# @Author : 茶葫芦
# @Site   : 
# @File   : Regularization.py

#添加多项式项,线性方式对非线性进行拟合
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


if __name__ == '__main__':
    x =np.random.randint(1,100,size=50)
    X =np.array([x])
    y = 4.0 * X **2 +30 *X +90 +np.random.normal(size =50)
    # x_train,x_test,y_train,y_test = train_test_split(X,y)
    poly =PolynomialFeatures(degree=2)
    lr=LinearRegression()
    poly.fit(X)
    x2 = poly.transform(X)
    # x_test2 =poly.transform(x_test)
    # lr.fit(x_train2,y_train)
    # res =lr.score(x_test2,y_test)
    # print(res)

    poly_reg = Pipeline([
        ("poly",PolynomialFeatures(degree =2)),
        ("std_scaler",StandardScaler()),
        ("lin_reg",LinearRegression())
    ])

    # poly_reg.fit(x_train, y_train)