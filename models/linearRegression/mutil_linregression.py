#!/usr/bin/env python
# encoding:utf-8
# @Time   : 2019/9/26
# @Author : 茶葫芦
# @Site   : 
# @File   : mutil_linregression.py
import numpy as np
from pub.metrics import r2_square

class lineregression():
    def __init__(self):
        self.a_ = None
        self.b_ = None
        self.threa = None

    def fit(self, x_train, y_train):
        x_b = np.hstack([np.ones((len(x_train),1)), x_train])
        self.threa = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y_train)
        self.b_, self.a_ = self.threa[0], self.threa[1:]
        return self

    def predict(self, x_predict):
        x_predict2 = np.hstack([np.ones((len(x_predict), 1)), x_predict])
        y_predict = self.threa.dot(x_predict2)
        return y_predict

    def score(self,x_test,y_test):
        return r2_square(y_test,self.predict(x_test))


if __name__ == '__main__':
    from sklearn.datasets import load_boston
    boston =load_boston()

    ln = lineregression()

    ln.fit(boston.data,boston.target)
    print(ln.a_,ln.b_)