# -*- coding: utf-8 -*-
"""
Created on Fri Oct 07 11:33:25 2016

@author: s6324900
"""

## Load data
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
paths =  "C:\Users\s6324900\Desktop\housing dataset"
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
    return rmse
    #print("RMSLE (of data): {:.3}".format(rmse))
    

# Compute R2 score
def r2_score(y,p):
    from sklearn.metrics import r2_score
    r2 = r2_score(y, p)
    return r2
    #print("R2 (on data): {:.2}".format(r2))

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
df2.drop(['YrSold','GarageYrBlt', 'MSSubClass'], axis = 1, inplace = True)
df2.head()

## View distribution of features using SNS
df2 = df2.fillna(df2.mean())
df3= df2.copy()

import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
sel_feature =  'MoSold'
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,8))  # plots on same row
sns.distplot(df2[sel_feature], kde = False, ax = ax1);
sns.regplot(x=df2[sel_feature], y= df2['SalePrice'], data=df2, ax= ax2);

df3[sel_feature] = np.where(df3[sel_feature] >=2450, 2450, df3[sel_feature])
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,8))  # plots on same row
sns.distplot(df3[sel_feature], kde = False, ax = ax1);
sns.regplot(x=df3[sel_feature], y= df3['SalePrice'], data=df3, ax= ax2);

# 'LotArea'>50,000 , cap at 50,000
# BsmtFinSF1 >2000
# MasVnrArea > 600
# TotalBsmtSF >2100, 1stFlrSF > 2450
##Iffy - LotArea
#Good feature - TotalBsmtSF, 1stFlrSF, GrLivArea, FullBath, BedroomAbvGr, TotRmsAbvGrd, GarageCars,GarageArea


## Caps
df3['1stFlrSF'] = np.where(df3['1stFlrSF'] >=2450, 2450, df3['1stFlrSF'])
df3['TotalBsmtSF'] = np.where(df3['TotalBsmtSF'] >=2100, 2100, df3['TotalBsmtSF'])
df3['LotArea'] = np.where(df3['LotArea'] >=50000, 50000, df3['LotArea'])
df3['BsmtFinSF1'] = np.where(df3['BsmtFinSF1'] >=2000, 2000, df3['BsmtFinSF1'])
df3['MasVnrArea'] = np.where(df3['MasVnrArea'] >=600, 600, df3['MasVnrArea'])

for x in range(1, df3.shape[0]+1):
    if df3.ix[x,'2ndFlrSF']== 0:
        df3.ix[x,'2flr_yn'] = 0
    else:
        df3.ix[x,'2flr_yn'] = 1
        
for x in range(1, df3.shape[0]+1):
    if df3.ix[x,'LowQualFinSF']== 0:
        df3.ix[x,'lowqsf_yn'] = 0
    else:
        df3.ix[x,'lowqsf_yn'] = 1

for x in range(1, df3.shape[0]+1):
    if df3.ix[x,'KitchenAbvGr']== 1:
        df3.ix[x,'kitgrade'] = 1
    else:
        df3.ix[x,'kitgrade'] = 0
        
for x in range(1, df3.shape[0]+1):
    if df3.ix[x,'MoSold']== 6 or df3.ix[x,'MoSold']== 7 :
        df3.ix[x,'summer_sale'] = 1
    else:
        df3.ix[x,'summer_sale'] = 0
        

df3.drop(['LotFrontage','OverallCond','BsmtFinSF2','LowQualFinSF', 'BsmtFullBath', 'BsmtHalfBath', 'HalfBath', 
          'KitchenAbvGr', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
          'MiscVal', 'MoSold' ,'SalePrice' ], axis = 1, inplace = True)
          
df3.shape
df3.head()

## TODO: Lets build a Linear regression on cats only
from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
y = df['SalePrice']
x = df3.copy()
kf = KFold(len(x), n_folds=5, shuffle=True, random_state=4)
p = np.zeros_like(y)
for train_index,test_index in kf:
    lr.fit(x.iloc[train_index], y.iloc[train_index])
    p[test_index] = lr.predict(x.iloc[test_index])

# Make sense of prediction
print(y.min(), y.max(), y.mean())
print(p.min(), p.max(), p.mean())
print(len(y), len(p))

r2_score(y,p)
#0.79
kaggle_rmse(y,p)
#0.338       

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
x = df3.copy()

kf = KFold(len(x), n_folds=5)
pred = np.zeros_like(y)
for train_index,test_index in kf:
        met.fit(x.iloc[train_index], y.iloc[train_index])
        pred[test_index] = met.predict(x.iloc[test_index])

print(y.min(), y.max(), y.mean())
print(pred.min(), pred.max(), pred.mean())

print('RMSLE on 5 folds:', kaggle_rmse(y,pred))
# 0.346
print('R2_score on 5 folds:', r2_score(y,pred))
#0.74

##Plot result
fig, ax = plt.subplots()
y = y
ax.scatter(y, pred, c='k')
ax.plot([-5,-1], [-5,-1], 'r-', lw=2)
ax.set_xlabel('Actual value')
ax.set_ylabel('Predicted value')
ax.set_ylim([0,800000])
ax.set_xlim([0,800000])
fig.savefig('Figure_10k_scatter_EN_l1_ratio.png')

pred_coef = met.coef_
for s in pred_coef:
    print(s)
# Print col_names
x = df3.copy()
for col in x.columns:
    print(col)

#Print mean of each column
for col in x.columns:
    print(x[col].mean())

met.intercept_

# TODO: Create the cat dataset
# Create a new dataframe with all non-object columns
df2_cat = df.copy()
col_list = []
for col in df2_cat.columns:
        if (df2_cat[col].dtypes == 'object'): # non-object columns
            col_list.append(col)

print(col_list)
print(len(col_list))
#43 catg columns

df2_cat = df2_cat.loc[:,col_list]
df2_cat.tail()
df2_cat.shape

# Add sale price to this dataset
df2_cat['SalePrice'] = df['SalePrice']

## Do some plotting 
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
sel_feature =  'GarageQual'
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,8))  # plots on same row
strip_plot = sns.stripplot(x= df2_cat[sel_feature], y= df2_cat['SalePrice'], data=df2_cat, jitter=True, ax=ax1);
bar_plot = sns.barplot(x= df2_cat[sel_feature], y= df2_cat['SalePrice'], data=df2_cat, ax=ax2);

        
#sns.swarmplot(x= df2_cat[sel_feature], y= df2_cat['SalePrice'], data=df2_cat);
#sns.violinplot(x= df2_cat[sel_feature], y= df2_cat['SalePrice'], data=df2_cat);

#sns.violinplot(x= df2_cat[sel_feature], y= df2_cat['SalePrice'], data=df2_cat, inner = None)
#sns.swarmplot(x= df2_cat[sel_feature], y= df2_cat['SalePrice'], data=df2_cat,  color="w", alpha=.5);

# 'LotShape' is Reg or not - then drop
# LotConfig = CulDSac - Y/n; then drop lotconfig
# Condition1 is Norm or not - then drop
# Exterior2nd is Vinyl or not - then drop
# Foundation is PConc or not - then drop
# BsmtFinType1 is GLQ or not - then drop 
# Electrical = SBrkr or not - then drop
# RoofStyle - Gable, Hip or Other - then drop
# GarageCond is TA, Fa or Other 
# if KitchenQual or FireplaceQu or GarageQual or 'ExterQual or 'BsmtQual = Ex or Fa
# SaleType = New or not - then drop
# SaleCondition is Partial Sale or not - then drop
# Neighbourhood is powerful but too many values; create 2/3 sub cats
# HouseStyle is powerful but too many values; create 2/3 sub cats

sel_list = ['MSZoning', 'LotShape', 'LotConfig', 'Neighborhood', 'Condition1', 'BldgType',
            'HouseStyle', 'RoofStyle', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond',
            'Foundation', 'BsmtQual', 'BsmtExposure', 'BsmtFinType1', 'HeatingQC', 'CentralAir',
            'Electrical', 'KitchenQual', 'GarageType','GarageFinish','GarageCond', 'PavedDrive',
            'SaleType', 'SaleCondition', 'GarageQual']
            
# Create a new reduced dataframe
df3_cat = df2_cat.loc[:,sel_list]        
df3_cat.shape
df3_cat.head()
# Further perform feature engineering 
for x in range(1, df3_cat.shape[0]+1):
    if df3_cat.ix[x,'LotShape']== 'Reg':
        df3_cat.ix[x,'reg_lot'] = 1
    else:
        df3_cat.ix[x,'reg_lot'] = 0
        
for x in range(1, df3_cat.shape[0]+1):
    if df3_cat.ix[x,'LotConfig']== 'CulDSac':
        df3_cat.ix[x,'culdsac'] = 1
    else:
        df3_cat.ix[x,'culdsac'] = 0
        
for x in range(1, df3_cat.shape[0]+1):
    if df3_cat.ix[x,'Condition1']== 'Norm':
        df3_cat.ix[x,'norm_cond'] = 1
    else:
        df3_cat.ix[x,'norm_cond'] = 0
        
for x in range(1, df3_cat.shape[0]+1):
    if df3_cat.ix[x,'RoofStyle']== 'Gable':
        df3_cat.ix[x,'roofst_new'] = 'Gable'
    elif df3_cat.ix[x,'RoofStyle']== 'Hip':
        df3_cat.ix[x,'roofst_new'] = 'Hip'
    else:
        df3_cat.ix[x,'roofst_new'] = 'Other'
        
for x in range(1, df3_cat.shape[0]+1):
    if df3_cat.ix[x,'Exterior2nd']== 'Vinyl':
        df3_cat.ix[x,'vinyl'] = 1
    else:
        df3_cat.ix[x,'vinyl'] = 0
        
for x in range(1, df3_cat.shape[0]+1):
    if df3_cat.ix[x,'ExterCond']== 'Fa':
        df3_cat.ix[x,'fa_ex_con'] = 1
    else:
        df3_cat.ix[x,'fa_ex_con'] = 0
        
        
for x in range(1, df3_cat.shape[0]+1):
    if df3_cat.ix[x,'Foundation']== 'PConc':
        df3_cat.ix[x,'found_conc'] = 1
    else:
        df3_cat.ix[x,'found_conc'] = 0
        
for x in range(1, df3_cat.shape[0]+1):
    if df3_cat.ix[x,'BsmtExposure']== 'Gd':
        df3_cat.ix[x,'bsmt_exp_gd'] = 1
    else:
        df3_cat.ix[x,'bsmt_exp_gd'] = 0
        
for x in range(1, df3_cat.shape[0]+1):
    if df3_cat.ix[x,'BsmtFinType1']== 'GLQ':
        df3_cat.ix[x,'bsmt_fin_glq'] = 1
    else:
        df3_cat.ix[x,'bsmt_fin_glq'] = 0
        
for x in range(1, df3_cat.shape[0]+1):
    if df3_cat.ix[x,'HeatingQC']== 'Ex':
        df3_cat.ix[x,'ex_heat_qc'] = 1
    else:
        df3_cat.ix[x,'ex_heat_qc'] = 0
        
for x in range(1, df3_cat.shape[0]+1):
    if df3_cat.ix[x,'CentralAir']== 'Y':
        df3_cat.ix[x,'central_air'] = 1
    else:
        df3_cat.ix[x,'central_air'] = 0
        
for x in range(1, df3_cat.shape[0]+1):
    if df3_cat.ix[x,'Electrical']== 'SBrkr':
        df3_cat.ix[x,'elec_sbkr'] = 1
    else:
        df3_cat.ix[x,'elec_sbkr'] = 0
        
        
for x in range(1, df3_cat.shape[0]+1):
    if df3_cat.ix[x,'KitchenQual']== 'Ex' or df3_cat.ix[x,'ExterQual']== 'Ex'or df3_cat.ix[x,'BsmtQual']== 'Ex' or df3_cat.ix[x,'GarageQual']== 'Ex':
        df3_cat.ix[x,'any_ex'] = 1
    else:
        df3_cat.ix[x,'any_ex'] = 0
        
        
for x in range(1, df3_cat.shape[0]+1):
    if df3_cat.ix[x,'KitchenQual']== 'Fa' or df3_cat.ix[x,'ExterQual']== 'Fa' or df3_cat.ix[x,'BsmtQual']== 'Fa'or df3_cat.ix[x,'GarageQual']== 'Fa'or df3_cat.ix[x,'GarageQual']== 'Po':
        df3_cat.ix[x,'any_po'] = 1
    else:
        df3_cat.ix[x,'any_po'] = 0
        
for x in range(1, df3_cat.shape[0]+1):
    if df3_cat.ix[x,'GarageType']== 'Attchd':
        df3_cat.ix[x,'gartype_new'] = 'Attchd'
    elif df3_cat.ix[x,'GarageType']== 'Detchd':
        df3_cat.ix[x,'gartype_new'] = 'Detchd'
    elif df3_cat.ix[x,'GarageType']== 'BuiltIn':
        df3_cat.ix[x,'gartype_new'] = 'Builtin'
    else:
        df3_cat.ix[x,'gartype_new'] = 'Other'
        
        
for x in range(1, df3_cat.shape[0]+1):
    if df3_cat.ix[x,'GarageCond']== 'TA':
        df3_cat.ix[x,'gar_cond'] = 'TA'
    elif df3_cat.ix[x,'GarageCond']== 'Fa':
        df3_cat.ix[x,'gar_cond'] = 'Fa'
    else:
        df3_cat.ix[x,'gar_cond'] = 'Other'

for x in range(1, df3_cat.shape[0]+1):
    if df3_cat.ix[x,'SaleType']== 'New':
        df3_cat.ix[x,'new_sale'] = 1
    else:
        df3_cat.ix[x,'new_sale'] = 0
        
for x in range(1, df3_cat.shape[0]+1):
    if df3_cat.ix[x,'SaleCondition']== 'Partial':
        df3_cat.ix[x,'partial_sale'] = 1
    else:
        df3_cat.ix[x,'partial_sale'] = 0
        
# Neighbourhood var
for x in range(1, df3_cat.shape[0]+1):
    if df3_cat.ix[x,'Neighborhood'] == 'NoRidge':
        df3_cat.ix[x,'nbr_cat'] = 'Best'
    elif df3_cat.ix[x,'Neighborhood'] == 'NridgHt':
        df3_cat.ix[x,'nbr_cat'] = 'Best'
    elif df3_cat.ix[x,'Neighborhood'] == 'StoneBr':
        df3_cat.ix[x,'nbr_cat'] = 'Best'
    elif df3_cat.ix[x,'Neighborhood'] == 'Timber':
        df3_cat.ix[x,'nbr_cat'] = 'Superior'
    elif df3_cat.ix[x,'Neighborhood'] == 'Veenker':
        df3_cat.ix[x,'nbr_cat'] = 'Superior'
    elif df3_cat.ix[x,'Neighborhood'] == 'Somerst':
        df3_cat.ix[x,'nbr_cat'] = 'Superior'
    elif df3_cat.ix[x,'Neighborhood'] == 'ClearCr':
        df3_cat.ix[x,'nbr_cat'] = 'Superior'
    elif df3_cat.ix[x,'Neighborhood'] == 'Crawfor':
        df3_cat.ix[x,'nbr_cat'] = 'Superior'        

    elif df3_cat.ix[x,'Neighborhood'] == 'CollgCr':
        df3_cat.ix[x,'nbr_cat'] = 'Above_Avg'
    elif df3_cat.ix[x,'Neighborhood'] == 'Blmngtn':
        df3_cat.ix[x,'nbr_cat'] = 'Above_Avg'
    elif df3_cat.ix[x,'Neighborhood'] == 'Gilbert':
        df3_cat.ix[x,'nbr_cat'] = 'Above_Avg'        
    elif df3_cat.ix[x,'Neighborhood'] == 'NWAmes':
        df3_cat.ix[x,'nbr_cat'] = 'Above_Avg'
    elif df3_cat.ix[x,'Neighborhood'] == 'SawyerW':
        df3_cat.ix[x,'nbr_cat'] = 'Above_Avg'
        
              
    elif df3_cat.ix[x,'Neighborhood'] == 'Mitchel':
        df3_cat.ix[x,'nbr_cat'] = 'Below_Avg'
    elif df3_cat.ix[x,'Neighborhood'] == 'NAmes':
        df3_cat.ix[x,'nbr_cat'] = 'Below_Avg'       
    elif df3_cat.ix[x,'Neighborhood'] == 'NPkVill':
        df3_cat.ix[x,'nbr_cat'] = 'Below_Avg'    
    elif df3_cat.ix[x,'Neighborhood'] == 'SWISU':
        df3_cat.ix[x,'nbr_cat'] = 'Below_Avg'    
    elif df3_cat.ix[x,'Neighborhood'] == 'Blueste':
        df3_cat.ix[x,'nbr_cat'] = 'Below_Avg' 
    elif df3_cat.ix[x,'Neighborhood'] == 'Sawyer':
        df3_cat.ix[x,'nbr_cat'] = 'Below_Avg'

        
    elif df3_cat.ix[x,'Neighborhood'] =='OldTown':
        df3_cat.ix[x,'nbr_cat'] = 'Inferior'
    elif df3_cat.ix[x,'Neighborhood'] =='Edwards':
        df3_cat.ix[x,'nbr_cat'] = 'Inferior'    
    elif df3_cat.ix[x,'Neighborhood'] =='BrkSide':
        df3_cat.ix[x,'nbr_cat'] = 'Inferior'  
    else: 
        df3_cat.ix[x,'nbr_cat'] = 'Worst'
        
# cats left -'MSZoning', 'BldgType', 'HouseStyle', 'roofst_new', 
#'MasVnrType',  'ExterQual' , 'BsmtQual' , 'KitchenQual' , 'gartype_new', 'GarageFinish'
# 'gar_cond', 'PavedDrive', 'nbr_cat'

df3_cat.shape        
# Perform drops        
df3_cat.drop(['LotShape','LotConfig','Condition1', 'RoofStyle' , 'Exterior2nd',
             'ExterCond','Foundation', 'BsmtExposure', 'BsmtFinType1', 
             'HeatingQC', 'CentralAir', 'Electrical', 'GarageType','GarageCond',
             'SaleType', 'SaleCondition', 'Neighborhood', 'GarageQual'
             ], axis = 1, inplace = True)

df3_cat.shape 
df4_cat = df3_cat.copy()
# 28 categories

df4_cat.head()
# Add to dummy all remain categories
           
df4_cat = pd.get_dummies(df4_cat)
df4_cat.head()       
df4_cat.describe()

for col in df4_cat.columns:
    print(col)

## TODO: Lets build a Linear regression on cats only
from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
y = df['SalePrice']
x = df4_cat.copy()
kf = KFold(len(x), n_folds=5, shuffle=True, random_state=4)
p = np.zeros_like(y)
for train_index,test_index in kf:
    lr.fit(x.iloc[train_index], y.iloc[train_index])
    p[test_index] = lr.predict(x.iloc[test_index])

# Make sense of prediction
print(y.min(), y.max(), y.mean())
print(p.min(), p.max(), p.mean())
print(len(y), len(p))

r2_score(y,p)
#0.76
kaggle_rmse(y,p)
#0.196        

### TODO: Now lets try ElasticNetCv

from sklearn.linear_model import ElasticNetCV
met = ElasticNetCV(l1_ratio=[.01, .05, .25, .5, .75, .95, .99])
y = df['SalePrice']
x = df4_cat.copy()

kf = KFold(len(x), n_folds=5)
pred = np.zeros_like(y)
for train_index,test_index in kf:
        met.fit(x.iloc[train_index], y.iloc[train_index])
        pred[test_index] = met.predict(x.iloc[test_index])

print(y.min(), y.max(), y.mean())
print(pred.min(), pred.max(), pred.mean())

print('RMSLE on 5 folds:', kaggle_rmse(y,pred))
# 0.224
print('R2_score on 5 folds:', r2_score(y,pred))
#0.673

##Plot result
fig, ax = plt.subplots()
y = y
ax.scatter(y, pred, c='k')
ax.plot([-5,-1], [-5,-1], 'r-', lw=2)
ax.set_xlabel('Actual value')
ax.set_ylabel('Predicted value')
ax.set_ylim([0,800000])
ax.set_xlim([0,800000])
fig.savefig('Figure_10k_scatter_EN_l1_ratio.png')

pred_coef = met.coef_
for s in pred_coef:
    print(s)
# Print col_names
x = df4_cat.copy()
for col in x.columns:
    print(col)

#Print mean of each column
for col in x.columns:
    print x[col].mean()

met.intercept_

## columnwise append the 2 datasets
df4_cat.drop(['SalePrice'
             ], axis = 1, inplace = True)
             
df3.drop(['SalePrice'
             ], axis = 1, inplace = True)

df_final = pd.concat([df4_cat, df3], axis=1, join_axes=[df4_cat.index])
df_final.shape

#Add target variable
df_final['SalePrice'] = df['SalePrice']
df_final.head()

#Pickle this dataset for future
paths = "C:\Users\s6324900\Desktop\housing dataset"
picfile = os.path.join(paths, "df.pkl")
df_final.to_pickle(picfile)

# To read dataset in future
df = pd.read_pickle(picfile)
df.shape