#!/usr/bin/env python
# encoding:utf-8
# @Time   : 2019/10/6
# @Author : 茶葫芦
# @Site   : 
# @File   : logistic.py
import numpy as np
from pub.metrics import accuracy

class logistic_regression():
    def __init__(self):
        self.a_ = None
        self.b_ = None
        self.threa = None

    def sigmoid(self,t):
        return 1./(1.+np.exp(-t))

    def fit_gradient(self, x_train, y_train,  eta=0.01, n_iters=1e4):



        def J(theta, x_b, y):
            # 损失函数
            y_hat = self.sigmoid(x_b.dot(theta))
            try:
                return - np.sum(y * np.log(y_hat) + (1- y)* np.log(1-y_hat)) / len(y)
            except:
                return float('inf')

        def dJ(theta, x_b, y):
            # 对theta求导
            return x_b.T.dot(self.sigmoid(x_b.dot(theta)) -y) / len(x_b)

        def gradient_descent(x_b, y, init_theta, eta, n_iters=1e4, epsilon=1e-8):
            theta = init_theta
            cur_iter = 0
            while cur_iter < n_iters:
                gradient = dJ(theta, x_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                if (abs(J(theta, x_b, y) - J(last_theta, x_b, y)) < epsilon):
                    break

                cur_iter += 1

            return theta

        x_b = np.hstack([np.ones((len(x_train), 1)), x_train])
        init_theta = np.zeros(x_b.shape[1])
        self._theta = gradient_descent(x_b,y_train,init_theta, eta)
        self.b_ = self._theta[0]
        self.a_ = self._theta[1:]

    def predict_prob(self, x_predict):
        x_predict2 = np.hstack([np.ones((len(x_predict), 1)), x_predict])
        y_predict = self.sigmoid(self.threa.dot(x_predict2))
        return y_predict


    def predict(self, x_predict):
        x_predict2 = np.hstack([np.ones((len(x_predict), 1)), x_predict])
        y_predict = self.predict_prob(x_predict2)
        return np.array(y_predict >= 0.5,dtype='int')

    def score(self, x_test, y_test):
        return accuracy(y_test, self.predict(x_test))

