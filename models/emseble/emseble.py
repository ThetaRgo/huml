#!/usr/bin/env python
# encoding:utf-8
# @Time   : 2019/10/13
# @Author : 茶葫芦
# @Site   : 
# @File   : emseble.py

"""
集成学习的思路有voting(投票),bagging(取样),boosting(增强),stacking(叠模型)
"""

from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    x,y =make_moons(n_samples= 5000,noise=0.3,random_state=420)
    x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=666)
    voting_clf_hard =VotingClassifier(
        estimators=[
            ('log_clf',LogisticRegression()),
            ('svm_clf',SVC()),
            ('dt_clf',DecisionTreeClassifier(random_state=666))
        ],voting='hard'
    )

    voting_clf_soft =VotingClassifier(
        estimators=[
            ('log_clf',LogisticRegression()),
            ('svm_clf',SVC(probability=True)),
            ('dt_clf',DecisionTreeClassifier(random_state=666))
        ],voting='soft'
    )

    voting_clf_hard.fit(x_train,y_train)
    voting_clf_soft.fit(x_train,y_train)
    print("-----voting hard------")
    print(voting_clf_hard.score(x_test,y_test))
    print("-----voting soft------")
    print(voting_clf_soft.score(x_test,y_test))


    bagging_clf =BaggingClassifier(
        DecisionTreeClassifier(),
        n_estimators= 500,
        max_samples=100,
        bootstrap=True,
        oob_score=True,
        n_jobs= -1
    )
    bagging_clf.fit(x_train,y_train)
    print("-----bagging oob score------")
    print(bagging_clf.oob_score_)

    rf_clf =RandomForestClassifier(n_estimators=500,
                                   oob_score=True,
                                   n_jobs= -1
                                   )
    rf_clf.fit(x_train,y_train)
    print("-----forest oob score------")
    print(rf_clf.oob_score_)

    ada_clf =AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
                                n_estimators=500,
                                )
    ada_clf.fit(x_train,y_train)
    print("-----adaboosting oob score------")
    print(ada_clf.score(x_test,y_test))

    # 注意 GradientBoostingClassifier 只能选择决策树模型作为训练模型
    gb_clf =GradientBoostingClassifier(max_depth=2,n_estimators=500)
    gb_clf.fit(x_train,y_train)
    print("-----GradientBoostingClassifier oob score------")
    print(gb_clf.score(x_test,y_test))