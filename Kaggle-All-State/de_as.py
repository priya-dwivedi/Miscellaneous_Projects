# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 16:14:02 2016

@author: s6324900
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
import os
paths =  "C:\Users\s6324900\Desktop\insurance"
filename = os.path.join(paths, "train.csv")

df =  pd.DataFrame.from_csv(filename)
df.head()
df.shape

#Define Kaggle Evaluation Metrics
# If predicted is negative, then predicted = 0 to allow for taking log
def kaggle_mae(true, pred):
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(true,pred)
    return mae
    #print("RMSLE (of data): {:.3}".format(rmse))
    
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

## View distribution of features using SNS
df2 = df2.fillna(df2.mean())
df3= df2.copy()

df3['log_loss'] = np.log1p(df3['loss'])
df3.describe()
## Lets first look at the loss function

sns.distplot(df3['loss'], kde = False);
sns.boxplot(df3['loss']);

sns.distplot(df3['log_loss'], kde = False);
sns.boxplot(df3['log_loss']);

## Loss was very left skewed. Taking log1p makes it much much better

sel_feature =  'cont7'
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,8))  # plots on same row
sns.distplot(df3[sel_feature], kde = False, ax = ax1);
sns.regplot(x=df3[sel_feature], y= df3['log_loss'], data=df3, ax= ax2);

# cont 1 - pretty flat across loss; dist continuous from 0 to 1. Dist is normal
# cont 2 - useful; dist discrete from 0 to 1; presence of outliers
# cont 3 - pretty flat across loss; dist continuous from 0 to 1
# cont 4 - pretty flat across loss; dist continuous from 0 to 1
## Recommend no changes at this point

## Now lets look at Categorical data
df2_cat = df.copy()
col_list = []
for col in df2_cat.columns:
        if (df2_cat[col].dtypes == 'object'): # non-object columns
            col_list.append(col)

print(col_list)

df2_cat = df2_cat.loc[:,col_list]
df2_cat.tail()
df2_cat.shape

# Add Log loss
df2_cat['log_loss'] = df3['log_loss']

## cat 89- 116 have multiple columns. those need to be pared down
# Use seaborn

sel_feature =  'cat116'
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,8))  # plots on same row
strip_plot = sns.stripplot(x= df2_cat[sel_feature], y= df2_cat['log_loss'], data=df2_cat, jitter=True, ax=ax1);
bar_plot = sns.barplot(x= df2_cat[sel_feature], y= df2_cat['log_loss'], data=df2_cat, ax=ax2);

# Pare down cats
df2_cat.ix[df2_cat.cat89.isin(['E', 'H','I', 'G']), 'cat89'] = 'Other'
df2_cat.ix[df2_cat.cat90.isin(['F', 'E','G']), 'cat90'] = 'Other'
df2_cat.ix[df2_cat.cat92.isin(['F', 'D','I']), 'cat92'] = 'Other'
df2_cat.ix[df2_cat.cat94.isin(['F', 'A','E', 'G']), 'cat94'] = 'Other'
df2_cat.ix[df2_cat.cat96.isin(['I', 'C']), 'cat96'] = 'Other'
df2_cat.ix[df2_cat.cat97.isin(['F', 'B']), 'cat97'] = 'Other'
df2_cat.ix[df2_cat.cat99.isin(['M', 'H', 'G','I','O']), 'cat99'] = 'Other'
df2_cat.ix[df2_cat.cat101.isin(['E', 'H', 'N','B','U', 'K']), 'cat101'] = 'Other'
df2_cat.ix[df2_cat.cat102.isin(['G', 'F']), 'cat102'] = 'Other_norm'
df2_cat.ix[df2_cat.cat102.isin(['H', 'J']), 'cat102'] = 'Other_high'
df2_cat.ix[df2_cat.cat103.isin(['L', 'K', 'N','J']), 'cat103'] = 'Other'
df2_cat.ix[df2_cat.cat104.isin(['N', 'O', 'B','A', 'Q']), 'cat104'] = 'Other'
df2_cat.ix[df2_cat.cat105.isin(['P', 'Q', 'R','O','S']), 'cat105'] = 'Other'
df2_cat.ix[df2_cat.cat106.isin(['N', 'O', 'R','B']), 'cat106'] = 'Other'
df2_cat.ix[df2_cat.cat107.isin(['P', 'U', 'R','B', 'S']), 'cat107'] = 'Other'
df2_cat.ix[df2_cat.cat111.isin(['F', 'B', 'Y','D']), 'cat111'] = 'Other'
df2_cat.ix[df2_cat.cat114.isin(['V', 'D', 'X','W', 'S', 'G']), 'cat114'] = 'Other'

# no change - cat 91, cat93, cat95, cat98, cat 100, cat108
# Drop if var is not useful and has too many values
df2_cat.drop(['cat110', 'cat109', 'cat112', 'cat116', 'cat115', 'cat113',  
             ], axis = 1, inplace = True)

df2_cat.shape 
df3_cat = df2_cat.copy()
# 111 categories

df3_cat.head()
# Add to dummy all remain categories
           
df3_cat = pd.get_dummies(df3_cat)
df3_cat.head()       
#415 columns!
df3_cat.describe()


## columnwise append the 2 datasets
df3_cat.drop(['log_loss'
             ], axis = 1, inplace = True)
             
df3.drop(['loss'
             ], axis = 1, inplace = True)

df_final = pd.concat([df3_cat, df3], axis=1, join_axes=[df3_cat.index])
df_final.shape

df_final.head()

#Pickle this dataset for future
paths = "C:\Users\s6324900\Desktop\insurance"
picfile = os.path.join(paths, "df_train.pkl")
df_final.to_pickle(picfile)

# To read dataset in future
paths = "C:\Users\s6324900\Desktop\insurance"
picfile = os.path.join(paths, "df_train.pkl")
df = pd.read_pickle(picfile)
df.shape

## Define labels and features from training dataset
from sklearn.cross_validation import train_test_split
labels = df['log_loss']

features = df.copy()
features.drop(['log_loss'], axis = 1, inplace = True)
print(labels.shape, features.shape)

X_train, X_valid, y_train, y_valid = train_test_split(features, labels, test_size=0.20, random_state=12)

y_valid_exp = np.expm1(y_valid)
# Success
print("Training set", X_train.shape, y_train.shape)
print("Validation set", X_valid.shape, y_valid.shape)

## Lets do Lasso Regression for feature selection
from sklearn import linear_model
met = linear_model.LassoCV(alphas=[0.1,0.5, 1.0, 10.0, 15.0, 20.0, 50.0])
met.fit(X_train, y_train)
pred_valid_log = met.predict(X_valid)

pred_valid_exp = np.expm1(pred_valid_log)

# Best alpha
print("Best alpha:" ,met.alpha_)
pred_lasso = pred_valid_exp

kaggle_mae(y_valid_exp, pred_lasso)
#1690

coef = pd.Series(met.coef_, index = X_train.columns)
coef.to_csv("C:\Users\s6324900\Desktop\insurance\lasso_coef.csv")
# export to csv - only 2 coeffs were positive 80_b, 80_d

# Use sklearn feature selection
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel

clf = ExtraTreesRegressor()
clf = clf.fit(X_train, y_train)
clf.feature_importances_  

importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

## Select top 40 features
model = SelectFromModel(clf, threshold = 0.004)
model.fit(X_train, y_train)
X_new = model.transform(X_train)
n_features = X_new.shape 
print(n_features) 
X_new.head()
 
X_valid_new = model.transform(X_valid)
X_valid_new.shape

## Linear Regression on overall data and on selected sub sample
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

# Train the model using the training sets
lr.fit(X_train, y_train)
pred_valid_log = lr.predict(X_valid)

pred_valid_exp = np.expm1(pred_valid_log)
pred_linear = pred_valid_exp

kaggle_mae(y_valid_exp, pred_linear)
# 1267.83

# Linear Regression on sub sample
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

# Train the model using the training sets
lr.fit(X_new, y_train)
pred_valid_log = lr.predict(X_valid_new)

pred_valid_exp = np.expm1(pred_valid_log)
pred_linear = pred_valid_exp


kaggle_mae(y_valid_exp, pred_linear)
#1311
## Plots

print(y_valid_exp.min(), y_valid_exp.max(), y_valid_exp.mean())
print(pred_linear.min(), pred_linear.max(), pred_linear.mean())

fig, ax = plt.subplots()
ax.scatter(y_valid_exp, pred_linear, c='k')
ax.plot([-5,-1], [-5,-1], 'r-', lw=2)
ax.set_ylabel('Predicted value')
ax.set_ylim([0, 50000])
ax.set_xlim([0,50000])


## Lets try ridge regression
from sklearn import linear_model
met = linear_model.RidgeCV(alphas=[0.1,0.5, 1.0, 10.0, 15.0, 20.0, 50.0])
met.fit(X_new, y_train)
pred_valid_log = met.predict(X_valid_new)

pred_valid_exp = np.expm1(pred_valid_log)
pred_ridge = pred_valid_exp

# Best alpha
print("Best alpha:" ,met.alpha_)

kaggle_mae(y_valid_exp, pred_ridge)
#1311 - Defaults to OLS


## Lets try decision tree regressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.grid_search import GridSearchCV
met = GridSearchCV(DecisionTreeRegressor(random_state = 27), cv=5,
                   param_grid={"max_depth": [5,10,15,20,25]
                              }, scoring = 'mean_absolute_error' )

met.fit(X_new, y_train)
pred_valid_log = met.predict(X_valid_new)

print(met.best_estimator_)

pred_valid_exp = np.expm1(pred_valid_log)
pred_dtr = pred_valid_exp

kaggle_mae(y_valid_exp, pred_dtr)
#1324
## DTR on full data

from sklearn.tree import DecisionTreeRegressor
from sklearn.grid_search import GridSearchCV
met = GridSearchCV(DecisionTreeRegressor(random_state = 27), cv=5,
                   param_grid={"max_depth": [5,10,15,20,25]
                              }, scoring = 'mean_absolute_error' )

met.fit(X_train, y_train)
pred_valid_log = met.predict(X_valid)
print(met.best_estimator_)

pred_valid_exp = np.expm1(pred_valid_log)
pred_dtr = pred_valid_exp

kaggle_mae(y_valid_exp, pred_dtr)
#1318 - Minimal benefit of using Full data vs top 50 samples

X_new[:5,:]


## Neural Net Regressor with GridSearch 

from sklearn.neural_network import MLPRegressor
from sklearn.grid_search import GridSearchCV
met = GridSearchCV(MLPRegressor(random_state = 27), cv=5,
                   param_grid={"activation": ['logistic', 'relu'],
                               "solver" : ['sgd', 'adam'],
                                "learning_rate_init" : [0.001, 0.05, 0.01, 0.5],
                              }, scoring = 'mean_absolute_error' )

met.fit(X_train, y_train)
pred_valid_log = met.predict(X_valid)
print(met.best_estimator_)

pred_valid_exp = np.expm1(pred_valid_log)
pred_nnet = pred_valid_exp

kaggle_mae(y_valid_exp, pred_nnet)