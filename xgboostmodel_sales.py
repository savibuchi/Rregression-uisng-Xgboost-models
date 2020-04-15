#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 21:58:39 2020

@author: savi
"""
import xgboost as xgb
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
#import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_log_error
#input_path = r'/Users/savi/Desktop/Women Data Science/'
Data = pd.read_csv('train.csv')
Data.head(10)

Data.info()

# 
Data['Competition_Metric'].isnull().sum()
# Shows 1764 null values

'''Let's plot this metric'''
Data['Competition_Metric'].hist()

Data['Competition_Metric'].describe()

# Data imputaion using median 
Data['Competition_Metric']= Data['Competition_Metric'].fillna(Data['Competition_Metric'].median())


Data.info()

Data['Course_Domain'].value_counts()

Data['Course_Type'].value_counts()

Data['Short_Promotion'].value_counts()
Data['Long_Promotion'].value_counts()
Data['User_Traffic'].value_counts()

Data['Sales'].hist()

Data_new = Data.drop('ID',axis =1 )
Data_new = Data_new.drop('Course_ID',axis =1)

correlations = Data_new.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
names = ['']+list(Data_new)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()
correlations

# We can see sales is highly correlated with User_Traffic


y =Data_new.Sales
ID = Data.ID


Data_new= Data_new.loc[:, Data_new.columns != 'Sales']

one_hot1 = pd.get_dummies(Data_new['Course_Domain'])
# Drop column B as it is now encoded
Data_new = Data_new.drop('Course_Domain',axis = 1)
# Join the encoded df
Data_new = Data_new.join(one_hot1)

one_hot2 = pd.get_dummies(Data_new['Course_Type'])
# Drop column B as it is now encoded
Data_new = Data_new.drop('Course_Type',axis = 1)
Data_new = Data_new.join(one_hot2)

Data_new['Competition_Metric']= Data_new['Competition_Metric'].fillna(Data_new['Competition_Metric'].median())



X_train, X_test, y_train, y_test = train_test_split(Data_new, y, test_size=0.2)


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,tes
          'learning_rate': 0.01, 'loss': 'ls'}
clf = GradientBoostingRegressor(**params)


clf.fit(X_train, y_train)
mse = mean_squared_error(y_test, clf.predict(X_test))
mean_squared_log_error = mean_squared_log_error(y_test, clf.predict(X_test))
print("MSE: %.4f" % mse)

feat_importances = pd.Series(clf.feature_importances_, index=Data_new.columns)
feat_importances.nlargest(10).plot(kind='barh')




test = pd.read_csv('test_QkPvNLx.csv')
test.info()
test['Competition_Metric'].hist()
test['Competition_Metric'].mean()
test['Competition_Metric'].median()


test['Competition_Metric']= test['Competition_Metric'].fillna(test['Competition_Metric'].median())

ID_test = test.ID
one_hot3 = pd.get_dummies(test['Course_Domain'])
# Drop column B as it is now encoded
test = test.drop('Course_Domain',axis = 1)
# Join the encoded df
test = test.join(one_hot3)

one_hot4 = pd.get_dummies(test['Course_Type'])
# Drop column B as it is now encoded
test = test.drop('Course_Type',axis = 1)
test = test.drop('ID',axis =1 )
test = test.join(one_hot4)

test['Sales'] = clf.predict(test)

submission = pd.DataFrame({'ID':ID_test,'Sales':test['Sales']})

filename = 'submission.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)

