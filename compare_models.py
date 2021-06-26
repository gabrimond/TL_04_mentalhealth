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

#try this 
#https://stackoverflow.com/questions/42894871/how-to-plot-multiple-roc-curves-in-one-plot-with-legend-and-auc-scores-in-python/42895367
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

plt.figure(0).clf()

pred = np.random.rand(1000)
label = np.random.randint(2, size=1000)
fpr, tpr, thresh = metrics.roc_curve(label, pred)
auc = metrics.roc_auc_score(label, pred)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

pred = np.random.rand(1000)
label = np.random.randint(2, size=1000)
fpr, tpr, thresh = metrics.roc_curve(label, pred)
auc = metrics.roc_auc_score(label, pred)
plt.plot(fpr,tpr,label="data 2, auc="+str(auc))

plt.legend(loc=0)