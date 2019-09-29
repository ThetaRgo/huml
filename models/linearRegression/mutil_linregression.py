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
        """
        严谨数学公式求出的拟合参数
        :param x_train:
        :param y_train:
        :return:
        """
        x_b = np.hstack([np.ones((len(x_train), 1)), x_train])
        self.threa = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y_train)
        self.b_, self.a_ = self.threa[0], self.threa[1:]
        return self

    def fit_gradient(self, x_train, y_train,  eta=0.01, n_iters=1e4):
        def J(theta, x_b, y):
            # 线性回归损失函数
            return np.sum((x_b.dot(theta) - y) ** 2) / len(y)

        def dJ(theta, x_b, y):
            # 对theta求导
            res = np.empty(len(theta))
            res[0] = np.sum(x_b.dot(theta) - y)
            for i in range(1, len(theta)):
                res[i] = (x_b.dot(theta) - y).dot(x_b[:, i])

            return res * 2 / len(x_b)

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

    def fit_sgd(self,x_train,y_train,n_iters=5, t0=5, t1=50):
        """随机梯度下降"""
        def dJ_sgd(theta,x_b_i,y_i):
            return x_b_i * (x_b_i.dot(theta) - y_i) * 2.

        def sgd(X_b, y, initial_theta, n_iters, t0=5, t1=50):

            def learning_rate(t):
                """模拟退火算学习率"""
                return t0 / (t + t1)

            theta = initial_theta
            m = len(X_b)

            for cur_iter in range(n_iters):
                indexes = np.random.permutation(m)
                X_b_new = X_b[indexes]
                y_new = y[indexes]
                for i in range(m):
                    gradient = dJ_sgd(theta, X_b_new[i], y_new[i])
                    theta = theta - learning_rate(cur_iter * m + i) * gradient

            return theta

        X_b = np.hstack([np.ones((len(x_train), 1)), x_train])
        initial_theta = np.random.randn(X_b.shape[1])
        self._theta = sgd(X_b, y_train, initial_theta, n_iters, t0, t1)

        self.b_ = self._theta[0]
        self.a_ = self._theta[1:]

        return self


    def predict(self, x_predict):
        x_predict2 = np.hstack([np.ones((len(x_predict), 1)), x_predict])
        y_predict = self.threa.dot(x_predict2)
        return y_predict

    def score(self, x_test, y_test):
        return r2_square(y_test, self.predict(x_test))


if __name__ == '__main__':
    from sklearn.datasets import load_boston
    from sklearn.preprocessing import StandardScaler
    boston = load_boston()
    ln = lineregression()
    ss =StandardScaler()
    data=boston.data
    y=boston.target
    ss.fit(boston.data)
    data = ss.transform(data)
    ln.fit_sgd(data, boston.target,n_iters= 20)
    print(ln.a_, ln.b_)
