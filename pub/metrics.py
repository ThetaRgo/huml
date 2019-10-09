#!/usr/bin/env python
# encoding:utf-8
# @Time   : 2019/9/23
# @Author : 茶葫芦
# @Site   : 
# @File   : metrics.py
from numpy import dot,sqrt,abs,sum,var
import numpy as np


def accuracy(y_true, y_predict):
    """
    准确度
    """
    return sum(y_true == y_predict) / len(y_true)

def mse(y_true, y_predict):
    """均方误差"""
    return (y_true - y_predict).dot(y_true - y_predict) / len(y_true)

def rmse(y_true,y_predict):
    """均方根误差"""
    return sqrt(mse(y_true,y_predict))

def mae(y_true,y_predict):
    """平均绝对误差"""
    return sum(abs(y_true -y_predict))/len(y_true)

def r2_square(y_true,y_predict):
    """R方误差"""
    return 1 - mse(y_true,y_predict) / var(y_true)


def TN(y_test,y_predict):
    assert len(y_test) ==len(y_predict)
    return np.sum((y_test==0 & y_predict==0))

def TP(y_test,y_predict):
    assert len(y_test) ==len(y_predict)
    return np.sum((y_test==1 & y_predict==1))

def FN(y_test,y_predict):
    assert len(y_test) ==len(y_predict)
    return np.sum((y_test==1 & y_predict==0))

def FP(y_test,y_predict):
    assert len(y_test) ==len(y_predict)
    return np.sum((y_test==0 & y_predict==1))

def confusion_matrix(y_test,y_predict):
    return np.array([
        [TN(y_test,y_predict),FP(y_test,y_predict)],
        [FN(y_test, y_predict), TP(y_test, y_predict)]
    ])

def precision_score(y_test,y_predict):
    fp =FP(y_test,y_predict)
    tp =TP(y_test,y_predict)
    try:
        return tp / (fp+tp)
    except:
        return 0.0

def recall_score(y_test,y_predict):
    fn= FN(y_test,y_predict)
    tp =TP(y_test,y_predict)
    try:
        return tp / (tp + fn)
    except:
        return 0.0

def f1_score(y_test,y_predict):
    ps =precision_score(y_test,y_predict)
    rs =recall_score(y_test,y_predict)
    try:
        return 2 * ps * rs / (ps + rs)
    except:
        return 0.0

def FPR(y_test,y_predict):
    try:
        return FP(y_test,y_predict)/(TN(y_test,y_predict)+ FP(y_test,y_predict))
    except:
        return 0.0

def TPR(y_test,y_predict):
    try:
        return TP(y_test,y_predict) /(TP(y_test,y_predict) + FN(y_test,y_predict))
    except:
        return 0.0