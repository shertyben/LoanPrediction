# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 18:36:25 2019

@author: sudo
"""

## Importing Librairies
import pandas  as pd                   # For data manipulations
import numpy as np                     # For mathematical calculations 
import matplotlib.pyplot as plt        # For plotting graphs 
import seaborn as sns                  # For data visualization 


# Loadind datas to Data Frame
train=pd.read_csv("datas/train_u6lujuX_CVtuZ9i.csv") 
test=pd.read_csv("datas/test_Y3wMUE5_7gLdaTN.csv")


train_original=train.copy() 
test_original=test.copy()

train.columns

train.head(n=3)

train.shape, test.shape

train['Loan_Status'].value_counts() # Normalize can be set to True to print proportions instead of number 
train['Loan_Status'].value_counts(normalize=True)

# Plot Datas Loan_Status values
train['Loan_Status'].value_counts().plot.bar()


#Now lets visualize each variable separately. Different types of variables are Categorical, ordinal and numerical.
#
#Categorical features: These features have categories (Gender, Married, Self_Employed, Credit_History, Loan_Status)

#Ordinal features: Variables in categorical features having some order involved (Dependents, Education, Property_Area)

#Numerical features: These features have numerical values (ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term)

#Let’s visualize the categorical and ordinal features first.

# Independent Variable (Categorical)

plt.figure(1) 

plt.subplot(221) 
train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Gender') 

plt.subplot(222) 
train['Married'].value_counts(normalize=True).plot.bar(title= 'Married') 

plt.subplot(223) 
train['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self_Employed') 

plt.subplot(224) 
train['Credit_History'].value_counts(normalize=True).plot.bar(title= 'Credit_History') 

plt.show()


# Categorical Independent Variable vs Target Variable
Gender=pd.crosstab(train['Gender'],train['Loan_Status']) 
Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))


print ( 'Gender dtype : ' , Gender )





Married       =  pd.crosstab( train['Married'],train['Loan_Status'] ) 
Dependents    =  pd.crosstab( train['Dependents'],train['Loan_Status'] ) 
Education     =  pd.crosstab(train['Education'],train['Loan_Status']) 
Self_Employed =  pd.crosstab(train['Self_Employed'],train['Loan_Status']) 

Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.show() 


Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.show() 


Education.div(Education.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.show() 

Self_Employed.div(Self_Employed.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4)) 

plt.show()



# Numerical Independent Variable vs Target Variable
# We will try to find the mean income of people for which the loan has been approved vs the mean income of people for which the loan has not been approved.

train.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()




bins=[0,2500,4000,6000,81000] 
group=['Low','Average','High', 'Very high'] 
train['Income_bin']=pd.cut(train['ApplicantIncome'],bins,labels=group)
Income_bin=pd.crosstab(train['Income_bin'],train['Loan_Status']) 
Income_bin.div(Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.xlabel('ApplicantIncome') 
P = plt.ylabel('Percentage')


bins=[0,100,200,700] 
group=['Low','Average','High'] 
train['LoanAmount_bin']=pd.cut(train['LoanAmount'],bins,labels=group, include_lowest=True)
LoanAmount_bin=pd.crosstab(train['LoanAmount_bin'],train['Loan_Status']) 
LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.xlabel('LoanAmount') 
P = plt.ylabel('Percentage')

train.isnull().sum()

# We can consider these methods to fill the missing values:
## 1. For numerical variables: imputation using mean or median
## 2. For categorical variables: imputation using mode

train['Gender'].fillna( train['Gender'].mode()[0] , inplace=True )
train['Married'].fillna( train["Married"].mode()[0] , inplace=True )
train['Dependents'].fillna( train["Dependents"].mode()[0] , inplace=True )
train['Self_Employed'].fillna( train["Self_Employed"].mode()[0] , inplace=True )
train['Credit_History'].fillna( train['Credit_History'].mode()[0] , inplace=True )


train['Loan_Amount_Term'].value_counts()

train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)

train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

train.drop( columns=['LoanAmount_bin', 'Income_bin'], axis=1 , inplace=True)


train=train.drop('Loan_ID',axis=1) 
test=test.drop('Loan_ID',axis=1)

X = train.drop('Loan_Status',1) 
y = train.Loan_Status

X=pd.get_dummies(X) 
train=pd.get_dummies(train) 
test=pd.get_dummies(test)

from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.3)

from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
model = LogisticRegression() 
model.fit(x_train, y_train)



LogisticRegression( C=1.0, class_weight=None, dual=False, fit_intercept=True,          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,          penalty='l2', random_state=1, solver='liblinear', tol=0.0001,          verbose=0, warm_start=False )

# Here the C parameter represents inverse of regularization strength. Regularization is 
# applying a penalty to increasing the magnitude of parameter values in order to reduce 
# overfitting. Smaller values of C specify stronger regularization. To learn about other 
# parameters, refer here: 
# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html





