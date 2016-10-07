# -*- coding: utf-8 -*-
"""
Created on Fri Oct 07 15:03:29 2016

@author: s6324900
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
paths =  "C:\Users\s6324900\Desktop\housing dataset"
filename = os.path.join(paths, "test.csv")

df_test =  pd.DataFrame.from_csv(filename)
df_test.head()
df_test.shape


# Create a new dataframe with all non-object columns
df2 = df_test.copy()
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

for x in range(1461, df2.shape[0]+1461):
    if df2.ix[x,'YearBuilt'] == df2.ix[x,'YearRemodAdd']:
        df2.ix[x,'Remodeled'] = 0
    else:
        df2.ix[x,'Remodeled'] = 1
        
for x in range(1461, df2.shape[0]+1461):
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

## Caps
df3['1stFlrSF'] = np.where(df3['1stFlrSF'] >=2450, 2450, df3['1stFlrSF'])
df3['TotalBsmtSF'] = np.where(df3['TotalBsmtSF'] >=2100, 2100, df3['TotalBsmtSF'])
df3['LotArea'] = np.where(df3['LotArea'] >=50000, 50000, df3['LotArea'])
df3['BsmtFinSF1'] = np.where(df3['BsmtFinSF1'] >=2000, 2000, df3['BsmtFinSF1'])
df3['MasVnrArea'] = np.where(df3['MasVnrArea'] >=600, 600, df3['MasVnrArea'])

for x in range(1461, df3.shape[0]+1461):
    if df3.ix[x,'2ndFlrSF']== 0:
        df3.ix[x,'2flr_yn'] = 0
    else:
        df3.ix[x,'2flr_yn'] = 1
        
for x in range(1461, df3.shape[0]+1461):
    if df3.ix[x,'LowQualFinSF']== 0:
        df3.ix[x,'lowqsf_yn'] = 0
    else:
        df3.ix[x,'lowqsf_yn'] = 1

for x in range(1461, df3.shape[0]+1461):
    if df3.ix[x,'KitchenAbvGr']== 1:
        df3.ix[x,'kitgrade'] = 1
    else:
        df3.ix[x,'kitgrade'] = 0
        
for x in range(1461, df3.shape[0]+1461):
    if df3.ix[x,'MoSold']== 6 or df3.ix[x,'MoSold']== 7 :
        df3.ix[x,'summer_sale'] = 1
    else:
        df3.ix[x,'summer_sale'] = 0
        

df3.drop(['LotFrontage','OverallCond','BsmtFinSF2','LowQualFinSF', 'BsmtFullBath', 'BsmtHalfBath', 'HalfBath', 
          'KitchenAbvGr', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
          'MiscVal', 'MoSold'], axis = 1, inplace = True)
          
df3.shape
df3.head()
df3.tail()


# TODO: Create the cat dataset
# Create a new dataframe with all non-object columns
df2_cat = df_test.copy()
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
for x in range(1461, df3_cat.shape[0]+1461):
    if df3_cat.ix[x,'LotShape']== 'Reg':
        df3_cat.ix[x,'reg_lot'] = 1
    else:
        df3_cat.ix[x,'reg_lot'] = 0
        
for x in range(1461, df3_cat.shape[0]+1461):
    if df3_cat.ix[x,'LotConfig']== 'CulDSac':
        df3_cat.ix[x,'culdsac'] = 1
    else:
        df3_cat.ix[x,'culdsac'] = 0
        
for x in range(1461, df3_cat.shape[0]+1461):
    if df3_cat.ix[x,'Condition1']== 'Norm':
        df3_cat.ix[x,'norm_cond'] = 1
    else:
        df3_cat.ix[x,'norm_cond'] = 0
        
for x in range(1461, df3_cat.shape[0]+1461):
    if df3_cat.ix[x,'RoofStyle']== 'Gable':
        df3_cat.ix[x,'roofst_new'] = 'Gable'
    elif df3_cat.ix[x,'RoofStyle']== 'Hip':
        df3_cat.ix[x,'roofst_new'] = 'Hip'
    else:
        df3_cat.ix[x,'roofst_new'] = 'Other'
        
for x in range(1461, df3_cat.shape[0]+1461):
    if df3_cat.ix[x,'Exterior2nd']== 'Vinyl':
        df3_cat.ix[x,'vinyl'] = 1
    else:
        df3_cat.ix[x,'vinyl'] = 0
        
for x in range(1461, df3_cat.shape[0]+1461):
    if df3_cat.ix[x,'ExterCond']== 'Fa':
        df3_cat.ix[x,'fa_ex_con'] = 1
    else:
        df3_cat.ix[x,'fa_ex_con'] = 0
        
        
for x in range(1461, df3_cat.shape[0]+1461):
    if df3_cat.ix[x,'Foundation']== 'PConc':
        df3_cat.ix[x,'found_conc'] = 1
    else:
        df3_cat.ix[x,'found_conc'] = 0
        
for x in range(1461, df3_cat.shape[0]+1461):
    if df3_cat.ix[x,'BsmtExposure']== 'Gd':
        df3_cat.ix[x,'bsmt_exp_gd'] = 1
    else:
        df3_cat.ix[x,'bsmt_exp_gd'] = 0
        
for x in range(1461, df3_cat.shape[0]+1461):
    if df3_cat.ix[x,'BsmtFinType1']== 'GLQ':
        df3_cat.ix[x,'bsmt_fin_glq'] = 1
    else:
        df3_cat.ix[x,'bsmt_fin_glq'] = 0
        
for x in range(1461, df3_cat.shape[0]+1461):
    if df3_cat.ix[x,'HeatingQC']== 'Ex':
        df3_cat.ix[x,'ex_heat_qc'] = 1
    else:
        df3_cat.ix[x,'ex_heat_qc'] = 0
        
for x in range(1461, df3_cat.shape[0]+1461):
    if df3_cat.ix[x,'CentralAir']== 'Y':
        df3_cat.ix[x,'central_air'] = 1
    else:
        df3_cat.ix[x,'central_air'] = 0
        
for x in range(1461, df3_cat.shape[0]+1461):
    if df3_cat.ix[x,'Electrical']== 'SBrkr':
        df3_cat.ix[x,'elec_sbkr'] = 1
    else:
        df3_cat.ix[x,'elec_sbkr'] = 0
        
        
for x in range(1461, df3_cat.shape[0]+1461):
    if df3_cat.ix[x,'KitchenQual']== 'Ex' or df3_cat.ix[x,'ExterQual']== 'Ex'or df3_cat.ix[x,'BsmtQual']== 'Ex' or df3_cat.ix[x,'GarageQual']== 'Ex':
        df3_cat.ix[x,'any_ex'] = 1
    else:
        df3_cat.ix[x,'any_ex'] = 0
        
        
for x in range(1461, df3_cat.shape[0]+1461):
    if df3_cat.ix[x,'KitchenQual']== 'Fa' or df3_cat.ix[x,'ExterQual']== 'Fa' or df3_cat.ix[x,'BsmtQual']== 'Fa'or df3_cat.ix[x,'GarageQual']== 'Fa'or df3_cat.ix[x,'GarageQual']== 'Po':
        df3_cat.ix[x,'any_po'] = 1
    else:
        df3_cat.ix[x,'any_po'] = 0
        
for x in range(1461, df3_cat.shape[0]+1461):
    if df3_cat.ix[x,'GarageType']== 'Attchd':
        df3_cat.ix[x,'gartype_new'] = 'Attchd'
    elif df3_cat.ix[x,'GarageType']== 'Detchd':
        df3_cat.ix[x,'gartype_new'] = 'Detchd'
    elif df3_cat.ix[x,'GarageType']== 'BuiltIn':
        df3_cat.ix[x,'gartype_new'] = 'Builtin'
    else:
        df3_cat.ix[x,'gartype_new'] = 'Other'
        
        
for x in range(1461, df3_cat.shape[0]+1461):
    if df3_cat.ix[x,'GarageCond']== 'TA':
        df3_cat.ix[x,'gar_cond'] = 'TA'
    elif df3_cat.ix[x,'GarageCond']== 'Fa':
        df3_cat.ix[x,'gar_cond'] = 'Fa'
    else:
        df3_cat.ix[x,'gar_cond'] = 'Other'

for x in range(1461, df3_cat.shape[0]+1461):
    if df3_cat.ix[x,'SaleType']== 'New':
        df3_cat.ix[x,'new_sale'] = 1
    else:
        df3_cat.ix[x,'new_sale'] = 0
        
for x in range(1461, df3_cat.shape[0]+1461):
    if df3_cat.ix[x,'SaleCondition']== 'Partial':
        df3_cat.ix[x,'partial_sale'] = 1
    else:
        df3_cat.ix[x,'partial_sale'] = 0
        
# Neighbourhood var
for x in range(1461, df3_cat.shape[0]+1461):
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
# Add to dummy all remain categories - 70 cols here
           
df4_cat = pd.get_dummies(df4_cat)
df4_cat.head()       
df4_cat.describe()
df4_cat['HouseStyle_2.5Fin'] = 0 # This is the missing column

# Print col_names
x = df4_cat.copy()
for col in x.columns:
    print(col)

## columnwise append the 2 datasets
df_final_test = pd.concat([df4_cat, df3], axis=1, join_axes=[df4_cat.index])
df_final_test.shape


#Pickle this dataset for future
paths = "C:\Users\s6324900\Desktop\housing dataset"
picfile = os.path.join(paths, "df_test.pkl")
df_final_test.to_pickle(picfile)

# To read dataset in future
df_test_f = pd.read_pickle(picfile)
df_test_f.shape


## Best result so far
from sklearn import linear_model
met = linear_model.RidgeCV(alphas=[0.1,0.5, 1.0, 10.0, 15.0, 20.0, 50.0])
paths = "C:\Users\s6324900\Desktop\housing dataset"
picfile2 = os.path.join(paths, "df.pkl")

df = pd.read_pickle(picfile2)
df.shape
df.head()

y = df['SalePrice']
y = np.log(y)
x = x

from sklearn.cross_validation import KFold
kf = KFold(len(x), n_folds=5)
pred = np.zeros_like(y)
for train_index,test_index in kf:
        met.fit(x.iloc[train_index], y.iloc[train_index])
        pred[test_index] = met.predict(x.iloc[test_index])

print(y.min(), y.max(), y.mean())
print(pred.min(), pred.max(), pred.mean())

met.alpha_
print('RMSLE on 5 folds:', kaggle_rmse_log(y,pred))
#0.1314
print('R2_score on 5 folds:', r2_score(y,pred))
#0.891


paths = "C:\Users\s6324900\Desktop\housing dataset"
picfile1 = os.path.join(paths, "df_test.pkl")

# To read dataset in future
df_test_f = pd.read_pickle(picfile1)
df_test_f.shape

pred_test_log = met.predict(df_test_f)
pred_test_log.shape

pred_test_exp = np.exp(pred_test_log)
pred_test_exp[:5]
print(pred_test_exp)

np.savetxt("C:\Users\s6324900\Desktop\housing dataset\pred_kaggle.csv", pred_test_exp, delimiter=",")
