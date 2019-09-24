#!/usr/bin/env python
# encoding:utf-8
# @Time   : 2019/9/24
# @Author : 茶葫芦
# @Site   : 
# @File   : processing.py
import numpy as np
# 均值方差归一化
class meanStdStandardScaler():
    def __init__(self):
        self.mean_ = []
        self.std_ = []

    def fit(self,X):
        for col in range(X.shape[1]):
            self.mean_.append(np.mean(X[:,col]))
            self.std_.append(np.std(X[:,col]))

        return self

    def transform(self,X):
        res=np.empty(shape=X.shape,dtype=float)
        for col in range(X.shape[1]):
            res[:,col]=(X[:,col] - self.mean_[col])/self.std_[col]
        return res


if __name__ == '__main__':
    x=np.random.randint(0,100,(50,2))
    ms=meanStdStandardScaler()
    ms.fit(x)
    xstd=ms.transform(x)
    print(np.mean(xstd[:,0]),    np.mean(xstd[:,1]),    np.std(xstd[:,0]),    np.std(xstd[:,1]))