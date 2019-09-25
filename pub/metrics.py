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
