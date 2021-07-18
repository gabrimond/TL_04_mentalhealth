#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 13:41:19 2021

@author: Gabriele
"""

###COMPARE ALL ML ALGORITHMS ACCURACIES

model = np.array(['Logistic Regression','SVM','Decision Tree','Random Forest'])
from sklearn.metrics import accuracy_score
scores = np.array([accuracy_score(lgm_preds,y_test),accuracy_score(svm_pred, y_test),accuracy_score(dt_preds,y_test),accuracy_score(rf_preds,y_test)])
df = {'model': model, 'scores': scores}
sns.barplot(x='model',y='scores',data=df)

##next to the plot for accuracies, try to creat a AUC ROC plot containing all AUCs

from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

plt.figure(0).clf()

pred = np.random.rand(1000)
label = np.random.randint(2, size=1000)
fpr, tpr, thresh = metrics.roc_curve(y_test, logmodel.predict_proba(X_test_rfe)[:,1])
auc = metrics.roc_auc_score(y_test, logmodel.predict(X_test_rfe))
plt.plot(fpr,tpr,label="Logistic regression, auc="+str(auc))

pred = np.random.rand(1000)
label = np.random.randint(2, size=1000)
fpr, tpr, thresh = metrics.roc_curve(y_test, svm.decision_function((X_test_rfe)))
auc = metrics.roc_auc_score(y_test, svm.predict(X_test_rfe))
plt.plot(fpr,tpr,label="SVM, auc="+str(auc))

pred = np.random.rand(1000)
label = np.random.randint(2, size=1000)
fpr, tpr, thresh = metrics.roc_curve(y_test, dt.predict_proba(X_test_rfe)[:,1])
auc = metrics.roc_auc_score(y_test, dt.predict(X_test_rfe))
plt.plot(fpr,tpr,label="Decision Tree, auc="+str(auc))

pred = np.random.rand(1000)
label = np.random.randint(2, size=1000)
fpr, tpr, thresh = metrics.roc_curve(y_test, rf.predict_proba(X_test_rfe)[:,1])
auc = metrics.roc_auc_score(y_test, rf.predict(X_test_rfe))
plt.plot(fpr,tpr,label="Random Forest, auc="+str(auc))

plt.legend(loc=0)