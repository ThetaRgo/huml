#!/usr/bin/env python
# encoding:utf-8
# @Time   : 2019/9/23
# @Author : 茶葫芦
# @Site   : 
# @File   : model_selection.py

import numpy as np


def train_test_split(x, y, ratio=0.3, seed=None):
    """
    将x,y集合按照ratio比例分成训练测试数据集
    """
    assert len(x) == len(y), 'x,y must be the same length'
    assert ratio > 0 and ratio < 1, 'ratio must between 0 and 1'
    if seed:
        np.random.seed(seed)

    shuffle_indeies = np.random.permutation(len(x))
    train_indies = int(len(x) * ratio)
    train_x = x[shuffle_indeies[:train_indies]]
    train_y = y[shuffle_indeies[:train_indies]]
    test_x = x[shuffle_indeies[train_indies:]]
    test_y = y[shuffle_indeies[train_indies:]]

    return train_x,train_y,test_x,test_y