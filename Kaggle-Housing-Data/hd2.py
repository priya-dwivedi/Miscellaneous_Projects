# -*- coding: utf-8 -*-
"""
Created on Tue Oct 04 10:47:18 2016

@author: s6324900
"""
## More intuition on non-object columns
## Note: RMSE is better with no outlier removak as the dataset is not normally distributed 
## Best RMSE is 0.215

## Created a dataset where I take log of most columns but that didnt work better 

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

df2.head()
df_mid = df2.fillna(0) # to test the effect of outlier removal
df3 = df2.fillna(0)

## Features are not normally distributed - Lets take log of 1+f
df3_log = df3.copy()
df3_log.drop(['yrs_since_built','yrs_since_remodel', 'yrs_since_garage_add', 'yrs_since_sol', 'Remodeled', 'garage_yn'], axis = 1, inplace = True)

df3_log = np.log(1+df3_log)
df3_log.head()

df3_log['yrs_since_built'] = df3['yrs_since_built']
df3_log['yrs_since_remodel'] = df3['yrs_since_remodel']
df3_log['yrs_since_garage_add'] = df3['yrs_since_garage_add']
df3_log['yrs_since_sol'] = df3['yrs_since_sol']
df3_log['Remodeled'] = df3['Remodeled']
df3_log['garage_yn'] = df3['garage_yn']

# Clean outliers
features = df3.keys()
features
for feature in features:
    # Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(df3[feature], 25)

    # Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(df3[feature], 75)

    # Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    #step = 1.5 * (Q3 - Q1)
    step = 4.0 * (Q3 - Q1)

    mean = np.mean(df3[feature])
    std = np.std(df3[feature])

    # Replace outliers with mean 
    df3[feature] = np.where(df3[feature] <=  mean - 3.0*std, mean - 3.0*std, df3[feature])
    df3[feature] = np.where(df3[feature] >= mean + 3*std, mean + 3*std, df3[feature])

# Compare original dataset with updated dataset
df2.describe()
df3.describe()
df3_log.describe()

##"Yes done!"

### TODO: Now lets try linear regression on cross fold validation

from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
y = df['SalePrice']
#x= df3.copy()
#x = df_mid.copy()
x = df3_log.copy()
kf = KFold(len(x), n_folds=5, indices=True, shuffle=True, random_state=4)
p = np.zeros_like(y)
for train_index,test_index in kf:
    lr.fit(x.iloc[train_index], y.iloc[train_index])
    p[test_index] = lr.predict(x.iloc[test_index])

# Make sense of prediction
print(y.min(), y.max(), y.mean())
print(p.min(), p.max(), p.mean())
print(len(y), len(p))

r2_score(y,p)
#0.81 #0.75 #0.75
kaggle_rmse(y,p)
#0.341 #0.453 #0.601

##Plot
fig, ax = plt.subplots()
y = y
pred = p
ax.scatter(y, pred, c='k')
ax.plot([-5,-1], [-5,-1], 'r-', lw=2)
ax.set_ylim([0,800000])
ax.set_xlim([0,800000])
ax.set_xlabel('Actual value')
ax.set_ylabel('Predicted value')
fig.savefig('Result of LR regression.png')


### TODO: Now lets try ElasticNetCv

from sklearn.linear_model import ElasticNetCV
met = ElasticNetCV(l1_ratio=[.01, .05, .25, .5, .75, .95, .99])
y = df['SalePrice']
x = df_mid.copy() #No outlier removal 
#x = df3.copy()
x = df3_log.copy()

kf = KFold(len(x), n_folds=5)
pred = np.zeros_like(y)
for train_index,test_index in kf:
        met.fit(x.iloc[train_index], y.iloc[train_index])
        pred[test_index] = met.predict(x.iloc[test_index])


print('RMSLE on 5 folds:', kaggle_rmse(y,pred))
# 0.346  #0.215 #0.28
print('R2_score on 5 folds:', r2_score(y,pred))
#0.74  #0.68

##Plot result
fig, ax = plt.subplots()
y = y
ax.scatter(y, pred, c='k')
ax.plot([-5,-1], [-5,-1], 'r-', lw=2)
ax.set_xlabel('Actual value')
ax.set_ylabel('Predicted value')
ax.set_ylim([0,1000000])
ax.set_xlim([0,1000000])
fig.savefig('Figure_10k_scatter_EN_l1_ratio.png')

print(y.min(), y.max(), y.mean())
print(pred.min(), pred.max(), pred.mean())

## See which vars are important
# predict coefficients
pred_coef = met.coef_
for s in pred_coef:
    print(s)
# Print col_names
x = df3.copy()
for col in x.columns:
    print(col)

#Print mean of each column
for col in x.columns:
    print x[col].mean()
    
## Top 5 most imp variables: TotalBsmtSF,1stFlrSF, GrLivArea, GarageArea, 2ndFlrSF
## Look at scatterplot
    
# Outlier removed dataset
plt.boxplot(df3['TotalBsmtSF'].values)
plt.scatter(df3['TotalBsmtSF'], df['SalePrice'], color='r')

#Original dataset
plt.boxplot(df2['TotalBsmtSF'].values)
plt.scatter(df2['TotalBsmtSF'], df['SalePrice'], color='r')

#Other boxplots
plt.boxplot(df2['GarageArea'].values)
plt.boxplot(df3_log['GarageArea'].values)

