# -*- coding: utf-8 -*-
"""
Created on Wed Oct 05 14:45:38 2016

@author: s6324900
"""

## Load data
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
paths = "C:\Users\s6324900\Desktop\housing dataset"
filename = os.path.join(paths, "train.csv")

df =  pd.DataFrame.from_csv(filename)
df.head()
df.shape

#Define Kaggle Evaluation Metrics
# If predicted is negative, then predicted = 0 to allow for taking log
def kaggle_rmse(y,p):
    from sklearn.metrics import mean_squared_error
    for x in range(0, len(p)):
            if p[x] < 0:
                p[x] = 0

    mse = mean_squared_error((np.log(1+y)), (np.log(1+p)))
    rmse = np.sqrt(mse)
    print("RMSLE (of data): {:.3}".format(rmse))
    

# Compute R2 score
def r2_score(y,p):
    from sklearn.metrics import r2_score
    r2 = r2_score(y, p)
    print("R2 (on data): {:.2}".format(r2))

## Identify all non object columns

# Create a new dataframe with all non-object columns
df2 = df.copy()
col_list = []
for col in df2.columns:
        if (df2[col].dtypes != 'object'): # non-object columns
            col_list.append(col)

print(col_list)

df2 = df2.loc[:,col_list]
df2.tail()
df2.shape

df2.dtypes
## Feature engineering
# Remove years and create new vars
df2['yrs_since_built'] = df2['YrSold'] - df2['YearBuilt']
df2['yrs_since_remodel']= df2['YrSold'] - df2['YearRemodAdd']
df2['yrs_since_garage_add']= df2['YrSold'] - df2['GarageYrBlt']
df2['yrs_since_sol'] = 2016-df2['YrSold']

for x in range(1, df2.shape[0]+1):
    if df2.ix[x,'YearBuilt'] == df2.ix[x,'YearRemodAdd']:
        df2.ix[x,'Remodeled'] = 0
    else:
        df2.ix[x,'Remodeled'] = 1
        
for x in range(1, df2.shape[0]+1):
    if df2.ix[x,'GarageYrBlt']== 'NaN':
        df2.ix[x,'garage_yn'] = 0
    else:
        df2.ix[x,'garage_yn'] = 1

df2.dtypes
df2.head()
## Vars that are like categorical - MSSubClass and target
df2.drop(['YrSold', 'YearBuilt', 'YearRemodAdd','GarageYrBlt', 'MSSubClass', 'SalePrice'], axis = 1, inplace = True)
df2['SalePrice'] = df['SalePrice']
df2.head()

## View distribution of features using SNS

import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
sel_feature =  'LotArea'
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,8))  # plots on same row
sns.distplot(df2[sel_feature], kde = False);

sns.jointplot(x=df2[sel_feature], y= df2['SalePrice'], data=df2);

bar_plot = sns.barplot(x= df2_cat[sel_feature], y= df2_cat['SalePrice'], data=df2_cat, ax=ax2);
