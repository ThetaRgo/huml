#!/usr/bin/env python
# encoding:utf-8
# @Time   : 2019/9/23
# @Author : 茶葫芦
# @Site   : 
# @File   : metrics.py


def accuracy(y_true,y_predict):
    """
    准确度
    """
    return sum(y_true==y_predict)/len(y_true)