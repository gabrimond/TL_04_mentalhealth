#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 12:14:27 2021

@author: Gabriele
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#increase length rows in output
pd.set_option('display.max_rows', None)

#change pands options to see all columns in the console
pd.set_option("display.max.columns", None)
#change to two decimal places (no need here but handy for in the future)
pd.set_option("display.precision", 2)

#%%

os.getcwd()
os.chdir('/Users/Gabriele/Desktop/TechLabs/mental_health')
os.getcwd()

clean_df = pd.read_csv('/Users/Gabriele/Desktop/TechLabs/mental_health/project/clean_df.csv')
clean_df.drop('Unnamed: 0', inplace=True, axis=1)
clean_df.rename(columns={'sought_mh_treatment': 'y'}, inplace=True)

#%%

cat_vars=['nr_employees', 'mh_benefits','difficulty_mh_leave', 
          'employer_mh_consequences', 'mh_issue_interview',
       'share_friendsfamily_mh', 'family_history', 'gender']

cont_vars=['age', 'y']

cont_clean_df=clean_df[cont_vars]
cat_clean_df=clean_df[cat_vars]
clean_df=cont_clean_df.join(pd.get_dummies(cat_clean_df))

#%%
clean_df.info()
###NOTE### Remove 'No' columns if you keep 'yes' to reduce redundancy in the categories.
clean_df.drop('mh_benefits_No', inplace=True, axis=1)
clean_df.drop('difficulty_mh_leave_Difficult', inplace=True, axis=1)
clean_df.drop('employer_mh_consequences_No', inplace=True, axis=1)
clean_df.drop('mh_issue_interview_No', inplace=True, axis=1)
clean_df.drop('share_friendsfamily_mh_Not open', inplace=True, axis=1)
clean_df.drop('family_history_No', inplace=True, axis=1)
clean_df.drop('gender_Female', inplace=True, axis=1)

#%%
X = clean_df.loc[:, clean_df.columns != 'y']
y = clean_df.loc[:, clean_df.columns == 'y']

#%%
#from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=31195)

#%%
#RFE from website
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


#how many and which features should we include to get the model with the highest accuracy?
#no of features
nof_list=np.arange(1,25)            
high_score=0
#Variable to store the optimum features
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 31195)
    logmodel=LogisticRegression()
    rfe = RFE(logmodel,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    logmodel.fit(X_train_rfe,y_train)
    score = logmodel.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
        
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))


#Run RFE after calculating how many features to include
cols = list(X.columns)
logmodel=LogisticRegression()
#Initializing RFE model
rfe = RFE(logmodel, 10)             
#Transforming data using RFE
X_rfe = rfe.fit_transform(X,y)  
#Fitting the data to model
logmodel.fit(X_rfe,y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)

#%%
import statsmodels.api as sm

y=clean_df.loc[:, clean_df.columns == 'y']
#X=clean_df.loc[:, clean_df.columns != 'y']

#BEFORE performing RFE selection
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())

#AFTER performing RFE selection
X=clean_df[selected_features_rfe]
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())

#%%
X_test_rfe=X_test[selected_features_rfe]
X_train_rfe=X_train[selected_features_rfe]

#train model
from sklearn.linear_model import LogisticRegression

logmodel=LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True , intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=1000, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)

logmodel.fit(X_train_rfe, y_train)

logmodel.score(X_test_rfe, y_test)
lgm_preds=logmodel.predict(X_test_rfe)

#%%
#evaluate the model
from sklearn.metrics import accuracy_score

accuracy_score(lgm_preds,y_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix= confusion_matrix(y_test, lgm_preds)
print(confusion_matrix)

#classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, lgm_preds))

#ROC curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

logit_roc_auc=roc_auc_score(y_test, logmodel.predict(X_test_rfe))
fpr, tpr, thresholds= roc_curve(y_test, logmodel.predict_proba(X_test_rfe)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area=%0.2f)' % logit_roc_auc)
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic Logistic regression')
plt.legend(loc="lower right")
#plt.savefig('Log_ROC')
plt.show()