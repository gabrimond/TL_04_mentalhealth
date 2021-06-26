#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 11:49:36 2021

@author: Gabriele
"""

import os

os.getcwd()
os.chdir('/Users/Gabriele/Desktop/TechLabs/mental_health/project/TL_04_mentalhealth')
os.getcwd()

#%%

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)

dt_preds = dt.predict(X_test)

#%%
#train svm with RFE features
dt.fit(X_train_rfe, y_train)

dt_preds=dt.predict(X_test_rfe)


#%%
from sklearn.metrics import accuracy_score

accuracy_score(dt_preds,y_test)

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(dt_preds, y_test))

#%%

#ROC curve

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

dt_roc_auc=roc_auc_score(y_test, dt.predict(X_test_rfe))
fpr, tpr, thresholds= roc_curve(y_test, dt.predict_proba(X_test_rfe)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='dt (area=%0.2f)' % dt_roc_auc)
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic Decision Tree')
plt.legend(loc="lower right")
#plt.savefig('DT_ROC')
plt.show()