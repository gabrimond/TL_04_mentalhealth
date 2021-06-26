#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 12:01:50 2021

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

clean_df = pd.read_csv('/Users/Gabriele/Desktop/TechLabs/mental_health/project/TL_04_mentalhealth/clean_df.csv')
clean_df.drop('Unnamed: 0', inplace=True, axis=1)
clean_df.rename(columns={'sought_mh_treatment': 'y'}, inplace=True)

#%%

for col in clean_df.columns:
    clean_df[col] = clean_df[col].astype('category')

#age to integer
clean_df['age'] = clean_df['age'].astype('int')

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
###NOTE### Remove 'No' columns if you keep 'yes'
clean_df.drop('mh_benefits_No', inplace=True, axis=1)
clean_df.drop('difficulty_mh_leave_Easy', inplace=True, axis=1)
clean_df.drop('employer_mh_consequences_No', inplace=True, axis=1)
clean_df.drop('mh_issue_interview_No', inplace=True, axis=1)
clean_df.drop('share_friendsfamily_mh_Not open', inplace=True, axis=1)
clean_df.drop('family_history_No', inplace=True, axis=1)
clean_df.drop('gender_Female', inplace=True, axis=1)

#%%

X = clean_df.loc[:, clean_df.columns != 'y']
y = clean_df.loc[:, clean_df.columns == 'y']

#%%
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)
#%%
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train,y_train)

rf_preds = rf.predict(X_test)

#%%
from sklearn.metrics import accuracy_score

accuracy_score(rf_preds,y_test)


