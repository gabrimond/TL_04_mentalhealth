#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 17:52:11 2021

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
#check and change current working directory
os.chdir('/Users/Gabriele/Desktop/TechLabs/mental_health/project/TL_04_mentalhealth')
os.getcwd()

osmi_2016 = pd.read_csv('/Users/Gabriele/Desktop/TechLabs/mental_health/OSMIdatasets/mental-heath-in-tech-2016_20161114.csv')

#%%
#modify your df (add, remove, mutate)
df = osmi_2016.copy()
df.shape

#rename columns
df = df.rename(
    columns={'What is your age?': 'age', 
             'What is your gender?':'gender',
             'Are you self-employed?':'self_employed',
             'How many employees does your company or organization have?':'nr_employees',
             'Is your employer primarily a tech company/organization?':'employer_tech',
             'Is your primary role within your company related to tech/IT?':'primaryrole_tech',
             'Does your employer provide mental health benefits as part of healthcare coverage?':'mh_benefits',
             'Do you know the options for mental health care available under your employer-provided coverage?':'know_mh_careoptions',
             'Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?':'employer_campaign_mh',
             'Does your employer offer resources to learn more about mental health concerns and options for seeking help?':'employer_info_mh',
             'Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?':'anonimity_protected_atwork',
             'If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:':'difficulty_mh_leave',
             'Do you think that discussing a mental health disorder with your employer would have negative consequences?':'employer_mh_consequences',
             'Do you think that discussing a physical health issue with your employer would have negative consequences?':'employer_ph_consequences',
             'Would you feel comfortable discussing a mental health disorder with your coworkers?':'comfortable_mh_colleagues',
             'Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?':'comfortable_mh_supervisor',
             'Do you feel that your employer takes mental health as seriously as physical health?':'employer_mh_as_ph',
             'Have you heard of or observed negative consequences for co-workers who have been open about mental health issues in your workplace?':'neg_consequences_mh_colleague',
             'Do you have medical coverage (private insurance or state-provided) which includes treatment of  mental health issues?': 'mh_med_coverage',
             'Do you know local or online resources to seek help for a mental health disorder?':'knowledge_resources_mh_help',
             'If you have been diagnosed or treated for a mental health disorder, do you ever reveal this to clients or business contacts?':'open_mh_clients',
             'If you have revealed a mental health issue to a client or business contact, do you believe this has impacted you negatively?':'impact_open_mh_clients',
             'If you have been diagnosed or treated for a mental health disorder, do you ever reveal this to coworkers or employees?':'open_mh_colleague_employer',
             'If you have revealed a mental health issue to a coworker or employee, do you believe this has impacted you negatively?':'impact_open_mh_coll_empl',
             'Do you believe your productivity is ever affected by a mental health issue?':'mh_affect_productivity',
             'If yes, what percentage of your work time (time performing primary or secondary job functions) is affected by a mental health issue?':'timeperc_mh_affect_productivity',
             'Do you have previous employers?':'prev_employers',
             'Have your previous employers provided mental health benefits?':'prev_employers_mh_benefits',
             'Were you aware of the options for mental health care provided by your previous employers?':'prev_employers_careoptions',
             'Did your previous employers ever formally discuss mental health (as part of a wellness campaign or other official communication)?':'prev_employers_campaign_mh',
             'Did your previous employers provide resources to learn more about mental health issues and how to seek help?':'prev_employers_info_mh',
             'Was your anonymity protected if you chose to take advantage of mental health or substance abuse treatment resources with previous employers?':'prev_employers_anonimity_protected',
             'Do you think that discussing a mental health disorder with previous employers would have negative consequences?':'prev_employers_mh_consequences',
             'Do you think that discussing a physical health issue with previous employers would have negative consequences?':'prev_employers_ph_consequences',
             'Would you have been willing to discuss a mental health issue with your previous co-workers?':'prev_employers_comfortable_mh_colleagues',
             'Would you have been willing to discuss a mental health issue with your direct supervisor(s)?':'prev_employers_comfortable_mh_supervisor',
             'Did you feel that your previous employers took mental health as seriously as physical health?':'prev_employers_mh_as_ph',
             'Did you hear of or observe negative consequences for co-workers with mental health issues in your previous workplaces?':'prev_employers_neg_consequences_mh_colleague',
             'Would you be willing to bring up a physical health issue with a potential employer in an interview?':'physical_issue_interview',
             'Would you bring up a mental health issue with a potential employer in an interview?':'mh_issue_interview',
             'Do you feel that being identified as a person with a mental health issue would hurt your career?':'mh_career_stigma',
             'Do you think that team members/co-workers would view you more negatively if they knew you suffered from a mental health issue?':'mh_colleague_stigma',
             'How willing would you be to share with friends and family that you have a mental illness?':'share_friendsfamily_mh',
             'Have you observed or experienced an unsupportive or badly handled response to a mental health issue in your current or previous workplace?':'unsupportive_mh_workplace',
             'Have your observations of how another individual who discussed a mental health disorder made you less likely to reveal a mental health issue yourself in your current workplace?':'workplaceexp_lesslikely_sharemh',
             'Do you have a family history of mental illness?':'family_history',
             'Have you had a mental health disorder in the past?':'mh_disorder_past',
             'Do you currently have a mental health disorder?':'mh_disorder_pres',
             'If yes, what condition(s) have you been diagnosed with?':'mh_yes_condition',
             'If maybe, what condition(s) do you believe you have?':'mh_maybe_condition',
             'Have you been diagnosed with a mental health condition by a medical professional?':'mh_profess_diagn',
             'If so, what condition(s) were you diagnosed with?':'mh_profess_diagn_condition',
             'Have you ever sought treatment for a mental health issue from a mental health professional?':'sought_mh_treatment',
             'If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?':'mh_interferes_work_yestreat',
             'If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?':'mh_interferes_work_notreat',
             'What country do you live in?':'country_live',
             'What US state or territory do you live in?':'us_live',
             'What country do you work in?':'country_work',
             'What US state or territory do you work in?':'us_work',
             'Which of the following best describes your work position?':'position_work',
             'Do you work remotely?':'remote_work'
             })

df.info()
#%%
#transform all variables to category
for col in df.columns:
    df[col] = df[col].astype('category')

#age to integer
df['age'] = df['age'].astype('int')

#%%
#clean gender
#revalue levels. Male and female. The remaining define themselves as queer or transgender. Category 'Other' would therefore include not only transgenders but all people who do not clearly identify with a cis gender. This can probably already influence menthal health. 
# at the same time, it is also importnat to realize that not all queer people (gay, lesbian etc.) would point that out when asked what their gender is. 
df['gender'].value_counts()
df['gender'].replace(['Male', 'Male ', 'male', 'male ', 'M', 'm', 'man', 'Male (cis)', 'Man', 'Cis Male', 'MALE', 'Cis male', 'cis male ', 'cis male', 'Malr', 'Dude', 'cisdude', 'mail', 'cis man', 'M|', 'Male.', "I'm a man why didn't you make this a drop down question. You should of asked sex? And I would of answered yes please. Seriously how much text can this take? "], 'Male', inplace=True)
df['gender'].replace(['Female', 'Female ', ' Female', 'female ', 'female', 'F', 'f', 'woman', 'Woman', 'I identify as female.', 'Female (props for making this a freeform field, though)', 'Cis female', 'Cis female ', 'Cis-woman', 'Cisgender Female', 'fm', 'female/woman', 'fem'], 'Female', inplace=True)

#I plan to create a value 'Other' for all the answers different from 'Male' and 'Female'. However, some answers are not really answers (e.g. 'human', 'unicorn')
#I think that these should be replaced with NAs, as they do not belong to any of the three categories. 
#df['gender'].replace(['Human', 'none of your business', 'Sex is male', 'Unicorn', 'human'], np.NaN, inplace=True)

#doubt: 'Female assigned at birth', 'AFAB' will now be replaced by Other. 
df['gender'].replace(['non-binary', 'Agender', 'Nonbinary', 'Genderfluid', 'Genderqueer', 'Genderflux demi-girl', 
                      'Genderfluid (born female)', 'Male (trans, FtM)', 'Female or Multi-Gender Femme', 
                      'Female assigned at birth ', 'Enby', 'Bigender', 'Androgynous', 'Fluid', 'Male/genderqueer',
                      'Other', 'Other/Transfeminine', 'Queer', 'Transgender woman', 'Transitioned, M2F', 
                      'female-bodied; no feelings about gender', 'genderqueer', 'genderqueer woman', 
                      'male 9:1 female, roughly', 'mtf', 'nb masculine', 'AFAB', 'Human', 'none of your business',
                      'Sex is male', 'Unicorn', 'human'], 'Other', inplace=True)
#%%
#clean age
#look for outliers
df.describe()
df['age'].median()
df[['age']].boxplot(whis=3)
sns.displot(df, x= 'age', binwidth=1)
#there are a number of outliers: 323, 3, and 99. I will replace these with the median of all the values. 
#This will have no consequences on the results because: 1) little number of outliers, 2) the median of the first two values is prob very close to the real value.
df['age']=df['age'].replace(to_replace=[323, 3, 99], value=df['age'].median())

#%%
#visualize features and their possible correlation with the label
sns.set(style='darkgrid')
sns.set_palette('hls', 3)
fig, ax=plt.subplots(figsize=(20,5))
ax=sns.countplot(x='self_employed', hue='sought_mh_treatment', data=df)

#to add percentages
for p in ax.patches:
    height=p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height+3,
            '{:1.2f}'.format(height/df.shape[0]),
            ha='center')

#first create a contingency table (or crosstab). These are used to summarise the relationship between categorical vars
cat_cor = pd.crosstab(index=df['self_employed'], columns=df['sought_mh_treatment'])
cat_cor

cat_cor.iloc[0].values

#chi2 test - add .iloc for each value within the category
from scipy import stats
(chi2, p, dof,_) = stats.chi2_contingency([cat_cor.iloc[0].values, cat_cor.iloc[1].values])
(chi2, p, dof,_) 

print('chi2     : ',chi2)
print('p-value  : ',p)
print('Degree of freedom  : ',dof)
#%%
df['self_employed'].value_counts()
sns.countplot(x='self_employed', hue='sought_mh_treatment', data=df)

#Since being self-employed does not seem to have a role in the classification of our outcome 
#(both from a data visualisation and from a statistical point of view), we can filter on self=employment.
#[This would also reduce the missingness]

#The above barplot suggests that self employment does not discriminate more on seeking for help as compared to people who are not self employed
df_notself_employed=df[df['self_employed']==0]
#%%
#to select the relevant features we look at:
    #1. content and overlap
    #2. data: visualisation and p-value (chisquare, ttest)
sns.countplot(x='nr_employees', hue='sought_mh_treatment', data=df_notself_employed)
sns.countplot(x='mh_benefits', hue='sought_mh_treatment', data=df_notself_employed)
sns.countplot(x='know_mh_careoptions', hue='sought_mh_treatment', data=df_notself_employed)
sns.countplot(x='employer_campaign_mh', hue='sought_mh_treatment', data=df_notself_employed)
sns.countplot(x='employer_info_mh', hue='sought_mh_treatment', data=df_notself_employed)
sns.countplot(x='anonimity_protected_atwork', hue='sought_mh_treatment', data=df_notself_employed)
sns.countplot(x='difficulty_mh_leave', hue='sought_mh_treatment', data=df_notself_employed)
sns.countplot(x='employer_mh_consequences', hue='sought_mh_treatment', data=df_notself_employed)
sns.countplot(x='employer_ph_consequences', hue='sought_mh_treatment', data=df_notself_employed)
sns.countplot(x='comfortable_mh_colleagues', hue='sought_mh_treatment', data=df_notself_employed)
sns.countplot(x='comfortable_mh_supervisor', hue='sought_mh_treatment', data=df_notself_employed)
sns.countplot(x='employer_mh_as_ph', hue='sought_mh_treatment', data=df_notself_employed)
sns.countplot(x='neg_consequences_mh_colleague', hue='sought_mh_treatment', data=df_notself_employed)
sns.countplot(x='mh_issue_interview', hue='sought_mh_treatment', data=df_notself_employed)
sns.countplot(x='workplaceexp_lesslikely_sharemh', hue='sought_mh_treatment', data=df_notself_employed)
sns.countplot(x='family_history', hue='sought_mh_treatment', data=df_notself_employed)
sns.countplot(x='mh_interferes_work_yestreat', hue='sought_mh_treatment', data=df_notself_employed)


#%%
#after selecting the relevant features we create a subset df
clean_df = df_notself_employed.iloc[:, np.r_[1,4,5,9,10,38,42,45,55,56,52]]
clean_df.info()

#visualise the missing data in the selected variables
clean_df.isnull()
sns.heatmap(clean_df.isnull())
#from the heatmap is clear that 'know_mh_careoptions' containes many missing values. To keep is simple, we can get rid of this column.
#Keep in mind, that in research this is not the best option. Often, missing values mean something and by removing participants with missing value
#you are actually selecting (selection bias) without knowing. This has huge impact on your results and above all your interpretation of the results. 
clean_df.drop(columns=['know_mh_careoptions'], axis=1, inplace=True)

clean_df.dtypes
clean_df.isnull().sum()
#we can see that the var 'gender' has 3 Nas. We remove the rows
clean_df.dropna(subset = ["gender"], inplace=True)

#reduce number of categories within two columns to simplify analysis (this should in reality not be done)
clean_df['difficulty_mh_leave'].value_counts()
clean_df['difficulty_mh_leave'].replace(['Somewhat easy', 'Very easy'], 'Easy', inplace=True)
clean_df['difficulty_mh_leave'].replace(['Somewhat difficult', 'Very difficult'], 'Difficult', inplace=True)


clean_df['share_friendsfamily_mh'].value_counts()
clean_df['share_friendsfamily_mh'].replace(['Somewhat open', 'Very open'], 'Open', inplace=True)
clean_df['share_friendsfamily_mh'].replace(['Somewhat not open', 'Not open at all'], 'Not open', inplace=True)
clean_df['share_friendsfamily_mh'].replace(['Not applicable to me (I do not have a mental illness)'], 'Not applicable (no mental illness)', inplace=True)

clean_df['mh_benefits'].value_counts()
clean_df['mh_benefits'].replace(['Not eligible for coverage / N/A'], 'Not eligible', inplace=True)

#clean_df.to_csv('clean_df.csv')