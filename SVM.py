#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 14:02:11 2021

@author: Gabriele
"""

from sklearn import svm

svm = svm.SVC(kernel='linear') # Linear Kernel
svm.fit(X_train, y_train)

svm_pred=svm.predict(X_test)

#%%
#train svm with RFE features
svm.fit(X_train_rfe, y_train)

svm_pred=svm.predict(X_test_rfe)

#%%
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(svm_pred, y_test))

#%%

#ROC curve

