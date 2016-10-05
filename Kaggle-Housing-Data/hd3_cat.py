# -*- coding: utf-8 -*-
"""
Created on Wed Oct 05 11:16:55 2016

@author: s6324900
"""
## Now lets look at object features

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
    if df3_cat.ix[x,'KitchenQual']== 'Fa' or df3_cat.ix[x,'ExterQual']== 'Fa' or df3_cat.ix[x,'BsmtQual']== 'Fa'or df3_cat.ix[x,'GarageQual']==(['Fa','Po']):
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
    if df3_cat.ix[x,'Neighborhood'] ==(['NoRidge','NridgHt','StoneBr']):
        df3_cat.ix[x,'nbr_cat'] = 'Best'
    elif df3_cat.ix[x,'Neighborhood'] == (['Timber','Veenker','Somerst', 'ClearCr', 'Crawfor']):
        df3_cat.ix[x,'nbr_cat'] = 'Superior'
    elif df3_cat.ix[x,'Neighborhood'] == (['CollgCr','Blmngtn', 'Gilbert', 'NWAmes', 'SawyerW']):
        df3_cat.ix[x,'nbr_cat'] = 'Above_Avg'
    elif df3_cat.ix[x,'Neighborhood'] == (['Mitchel','NAmes', 'NPkVill','SWISU', 'Blueste', 'Sawyer']):
        df3_cat.ix[x,'nbr_cat'] = 'Below_Avg'
    elif df3_cat.ix[x,'Neighborhood'] == (['OldTown','Edwards', 'BrkSide']):
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

## TODO: Lets build a Linear regression on cats only
from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
y = df['SalePrice']
x = df4_cat.copy()
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
#0.72
kaggle_rmse(y,p)
#0.213        

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
x = df4_cat.copy()

kf = KFold(len(x), n_folds=5)
pred = np.zeros_like(y)
for train_index,test_index in kf:
        met.fit(x.iloc[train_index], y.iloc[train_index])
        pred[test_index] = met.predict(x.iloc[test_index])

print(y.min(), y.max(), y.mean())
print(pred.min(), pred.max(), pred.mean())

print('RMSLE on 5 folds:', kaggle_rmse(y,pred))
# 0.233
print('R2_score on 5 folds:', r2_score(y,pred))
#0.64

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
