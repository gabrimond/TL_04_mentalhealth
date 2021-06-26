#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 14:02:11 2021

@author: Gabriele
"""

import os

os.getcwd()
os.chdir('/Users/Gabriele/Desktop/TechLabs/mental_health/project/TL_04_mentalhealth')
os.getcwd()

#%%

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

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

svm_roc_auc=roc_auc_score(y_test, svm.predict(X_test_rfe))
fpr, tpr, thresholds= roc_curve(y_test, svm.decision_function((X_test_rfe)))
plt.figure()
plt.plot(fpr, tpr, label='SVM (area=%0.2f)' % svm_roc_auc)
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic SVM')
plt.legend(loc="lower right")
#plt.savefig('SVM_ROC')
plt.show()