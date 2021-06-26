#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 12:01:50 2021

@author: Gabriele
"""

import os

os.getcwd()
os.chdir('/Users/Gabriele/Desktop/TechLabs/mental_health/project/TL_04_mentalhealth')
os.getcwd()

#%%

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train,y_train)

rf_preds = rf.predict(X_test)

#%%
#train svm with RFE features
rf.fit(X_train_rfe, y_train)

rf_preds=rf.predict(X_test_rfe)

#%%
from sklearn.metrics import accuracy_score

accuracy_score(rf_preds,y_test)

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(rf_preds, y_test))

#%%

#AUC

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

rf_roc_auc=roc_auc_score(y_test, rf.predict(X_test_rfe))
fpr, tpr, thresholds= roc_curve(y_test, rf.predict_proba(X_test_rfe)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='rf (area=%0.2f)' % rf_roc_auc)
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic Random Forest')
plt.legend(loc="lower right")
#plt.savefig('RF_ROC')
plt.show()
