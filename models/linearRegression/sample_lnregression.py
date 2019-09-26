#!/usr/bin/env python
# encoding:utf-8
# @Time   : 2019/9/24
# @Author : 茶葫芦
# @Site   : 
# @File   : sample_lnregression.py

import numpy as np
from pub.metrics import mse,rmse,mae,r2_square

class linearRegression():
    def __init__(self):
        self.a_ = None
        self.b = None

    def fit(self, x_train, y_train):
        """
        简单线性回归的最小化损失函数可以直接求解,只需按照
        训练样本求出对应的值就好
        :param x_train:
        :param y_train:
        :return:
        """
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        num1, num2 = 0.0, 0.0
        # for i in range(len(x_train)):
        #     num1 = +(x_train[i] - x_mean) * (y_train[i] - y_mean)
        #     num2 = +(x_train[i] - x_mean) ** 2
        num1 =np.dot((x_train - x_mean),(y_train -y_mean))
        num2 = np.dot((x_train - x_mean),(x_train -x_mean))

        self.a_ = num1 / num2
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        y_predict = [self._predict(x) for x in x_predict]
        return y_predict

    def _predict(self, x):
        return self.a_ * x + self.b_

    def score(self,y_test,y_predict):
        return r2_square(y_test,y_predict)


if __name__ == '__main__':
    x=np.random.randint(1000)
    y= 200 * x +19 +np.random.normal(1000)

    ln=linearRegression()


