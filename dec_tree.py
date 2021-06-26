#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 11:49:36 2021

@author: Gabriele
"""

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)

dt_preds = dt.predict(X_test)

#%%
from sklearn.metrics import accuracy_score

accuracy_score(dt_preds,y_test)

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(dt_preds, y_test))

#%%

#ROC curve
