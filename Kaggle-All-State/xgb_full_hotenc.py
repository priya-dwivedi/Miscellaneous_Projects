# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 13:05:44 2016

@author: priyankadwivedi
"""

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_absolute_error
from scipy.stats import skew, boxcox
from math import exp, log
#import xgboost as xgb


##Load the data
import os
paths =  "/Users/priyankadwivedi/Documents/Projects/Kaggle All State Insurance"


path_train = os.path.join(paths, "train.csv")
path_test= os.path.join(paths, "test.csv")

train_loader = pd.read_csv(path_train, dtype={'id': np.int32})
train = train_loader.drop(['id', 'loss'], axis=1)
test_loader = pd.read_csv(path_test, dtype={'id': np.int32})
test = test_loader.drop(['id'], axis=1)
ntrain = train.shape[0]
ntest = test.shape[0]
train_test = pd.concat((train, test)).reset_index(drop=True)
numeric_feats = train_test.dtypes[train_test.dtypes != "object"].index

train_test.shape
print(numeric_feats)

# Compute skew of features
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))
print("\nSkew in numeric features:")
print(skewed_feats)

# transform features with skew > 0.25 (this can be varied to find optimal value)
# transform features with skew > 0.25 (this can be varied to find optimal value)
skewed_feats = skewed_feats[skewed_feats > 0.25]
skewed_feats = skewed_feats.index # Just looks at cols with skew
print(skewed_feats)

train_test.describe()

## Apply log transformation
for feats in skewed_feats:
        train_test[feats] = train_test[feats] + 1
        train_test[feats] = np.log(train_test[feats])
train_test.describe()

#Identify categorical features
features = train.columns
cats = [feat for feat in features if 'cat' in feat]
print(cats)


import pandas as pd

#cat1 to cat116 have strings. The ML algorithms we are going to study require numberical data
#One-hot encoding converts an attribute to a binary vector

print(train['cat110'].unique())
#Variable to hold the list of variables for an attribute in the train and test data
labels = []

for feat in cats:
    train_range = train[feat].unique()
    test_range = test[feat].unique()
    labels.append(list(set(train_range) | set(test_range)))    

print(labels)
print(len(labels))
#Import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

## Create a dataset that only has categorical features
train_test_cat = train_test.loc[:,cats]
train_test_cat.head()

#One hot encode all categorical attributes
cats_list = []
split = len(labels)
for i in range(0, split):
    #Label encode
    label_encoder = LabelEncoder()
    label_encoder.fit(labels[i])
    feature = label_encoder.transform(train_test_cat.iloc[:,i])
    feature = feature.reshape(train_test_cat.shape[0], 1)
    #One hot encode
    onehot_encoder = OneHotEncoder(sparse=False,n_values=len(labels[i]))
    feature = onehot_encoder.fit_transform(feature)
    cats_list.append(feature)

# Make a 2D array from a list of 1D arrays
encoded_cats = np.column_stack(cats_list)

# Print the shape of the encoded data
print(encoded_cats.shape)
## Play around
encoded_cats[:5]

## Create a dataset of nums
train_test_num = train_test.loc[:,numeric_feats]
train_test_num.head()

##Free up space
del cats_list
del feature
del train_test_cat

#Concatenate encoded attributes with continuous attributes
dataset_encoded = np.concatenate((encoded_cats,train_test_num.values),axis=1)

del encoded_cats
del train_test_num
dataset_encoded.shape
dataset_encoded[:5]

# Split back into test and train
x_train = dataset_encoded[:ntrain, :]
x_test =  dataset_encoded[ntrain:, :]

print(x_train.shape, x_test.shape)

## Transform target into np log of loss
train_labels = np.log(np.array(train_loader['loss']))
train_ids = train_loader['id'].values.astype(np.int32)
test_ids = test_loader['id'].values.astype(np.int32)

## Aplit data to run Xgboost
from sklearn.cross_validation import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(x_train, train_labels, test_size=0.25, random_state=1981)

print("Training Data", X_train.shape, y_train.shape)
print("Validation Data", X_valid.shape, y_valid.shape)

## Now just try to run on previous chosen params

## Run with parameters identified by xgboost
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import mean_absolute_error
​
d_test = xgb.DMatrix(x_test)
early_stopping = 25
xgb_rounds = []
​
params = {}
params['booster'] = 'gbtree'
params['objective'] = "reg:linear"
params['eval_metric'] = 'mae'
#params['eta'] = 0.01
params['eta'] = 0.1228
params['gamma'] = 0.5668
params['min_child_weight'] = 2.534
params['colsample_bytree'] = 0.4769
params['subsample'] = 0.5445
params['max_depth'] = 6
params['max_delta_step'] = 0.0836
params['silent'] = 1
params['random_state'] = 929
​
d_train = xgb.DMatrix(X_train, label=y_train)
d_valid = xgb.DMatrix(X_valid, label=y_valid)
watchlist = [(d_train, 'train'), (d_valid, 'eval')]
    
    ## Build a model
clf = xgb.train(params,
                    d_train,
                    100000,
                    watchlist,
                    early_stopping_rounds=early_stopping)
​
    ## Evaluate model and predict
xgb_rounds.append(clf.best_iteration)
scores_val = clf.predict(d_valid, ntree_limit=clf.best_ntree_limit)
cv_score = mean_absolute_error(np.exp(y_valid), np.exp(scores_val))
print(' eval-MAE: %.6f' % cv_score)
y_pred_hot = np.exp(clf.predict(d_test, ntree_limit=clf.best_ntree_limit))
##1153 with non hot-enc data
## 1149 with hot-enc data but very very slow to run

solution = pd.DataFrame({"id":test_loader.id, "loss":y_pred_hot})
solution[:20]
solution.to_csv("/Users/priyankadwivedi/Documents/Projects/Kaggle All State Insurance/sub1_oct25.csv", index = False)

## LB score: 1130.5721

## Imports for XGboost tuning
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

## Tune using Bayesian Optimization - May help in improving score
#Only do 3 folds

from sklearn.gaussian_process import GaussianProcess
from scipy.optimize import minimize
from bayes_opt import BayesianOptimization

def xgboostcv(max_depth,
              learning_rate,
              n_estimators,
              gamma,
              min_child_weight,
              max_delta_step,
              subsample,
              colsample_bytree,
              silent=True,
              nthread=-1,
              seed = 422):
    return cross_val_score(xgb.XGBRegressor(max_depth=int(max_depth),
                                             learning_rate=learning_rate,
                                             n_estimators=int(n_estimators),
                                             silent=silent,
                                             nthread=nthread,
                                             gamma=gamma,
                                             min_child_weight=min_child_weight,
                                             max_delta_step=max_delta_step,
                                             subsample=subsample,
                                             colsample_bytree=colsample_bytree),
                           xte,
                           yte,
                           "neg_mean_absolute_error",
                           cv=5).mean()



## Break into several folds
max_val = 0
max_depth = 0
learning_rate = 0
gamma = 0
n_estimators = 0
min_child_weight = 0
subsample = 0
colsample_by_tree = 0
max_delta_step =0
xtrain = x_train
y = train_labels
n_folds = 3
i = 0
### Bayesian optimization on the train_less dataset
from sklearn.cross_validation import cross_val_score, cross_val_predict
folds = 3

kf = KFold(len(labels), n_folds=n_folds, shuffle=True)
for inTr, inTe in kf:
    ## Create datasets
    xtr = xtrain[inTr]
    ytr = y[inTr]
    xte = xtrain[inTe]
    yte = y[inTe]

    ### Pass these to xgboost
    xgboostBO = BayesianOptimization(xgboostcv,
                                         {'max_depth': (1, 11),
                                          'learning_rate': (0.01, 0.15),
                                          'n_estimators': (100, 1000),
                                          'gamma': (1., .01),
                                          'min_child_weight': (1, 10),
                                          'max_delta_step': (0, 0.1),
                                          'subsample': (0.1, 0.9),
                                          'colsample_bytree' :(0.1, 0.99)
                                          })

    #xgboostBO.maximize(init_points = 5, n_iter=15)
    xgboostBO.maximize(init_points = 15, n_iter=35)
    print('-'*53)
    print('fold', i+1)
    print('XGBOOST: %f' % xgboostBO.res['max']['max_val'])	
    print('XGBOOST: %f' % xgboostBO.res['max']['max_val'])	
    print('max_depth: %f' % xgboostBO.res['max']['max_params']['max_depth'])
    print('learning_rate: %f' % xgboostBO.res['max']['max_params']['learning_rate'])
    print('gamma: %f' % xgboostBO.res['max']['max_params']['gamma'])
    print('n_estimators: %f' % xgboostBO.res['max']['max_params']['n_estimators'])
    print('min_child_weight: %f' % xgboostBO.res['max']['max_params']['min_child_weight'])
    print('subsample: %f' % xgboostBO.res['max']['max_params']['subsample'])
    print('colsample_bytree: %f' % xgboostBO.res['max']['max_params']['colsample_bytree'])
    print('max_delta_step %f' % xgboostBO.res['max']['max_params']['max_delta_step'])

    max_val += xgboostBO.res['max']['max_val']
    max_depth += xgboostBO.res['max']['max_params']['max_depth']
    learning_rate += xgboostBO.res['max']['max_params']['learning_rate']
    gamma += xgboostBO.res['max']['max_params']['gamma']
    n_estimators += xgboostBO.res['max']['max_params']['n_estimators']
    min_child_weight += xgboostBO.res['max']['max_params']['min_child_weight']
    subsample += xgboostBO.res['max']['max_params']['subsample']
    colsample_by_tree += xgboostBO.res['max']['max_params']['colsample_bytree']
    max_delta_step += xgboostBO.res['max']['max_params']['max_delta_step']
    i += 1

max_val /= n_folds
max_depth /= n_folds
learning_rate /= n_folds
gamma /= n_folds
n_estimators /= n_folds
min_child_weight /= n_folds
subsample /= n_folds
colsample_by_tree /= n_folds
max_delta_step /= n_folds

print('Final Results') 
print('max value', max_val) 
print('max depth', max_depth) 
print('learning_rate', learning_rate) 
print('gamma', gamma) 
print('n_estimators', n_estimators) 
print('min_child_weight', min_child_weight) 
print('subsample', subsample) 
print('colsample_by_tree', colsample_by_tree)      
print('max_delta_step', max_delta_step)  

#max_depth	8.00527733
#lr	0.14693925
#gamma	0.7359492
#min_child_weight	6.3736338
#subsample	0.6310414
#colsample	0.5995254
#max_delta	0.086031


## Try xgboost on optimized parameters
from sklearn.cross_validation import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(x_train, train_labels, test_size=0.25, random_state=1981)

print("Training Data", X_train.shape, y_train.shape)
print("Validation Data", X_valid.shape, y_valid.shape)

## Now just try to run on previous chosen params

## Run with parameters identified by xgboost
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import mean_absolute_error
​
d_test = xgb.DMatrix(x_test)
early_stopping = 25
xgb_rounds = []
​
params = {}
params['booster'] = 'gbtree'
params['objective'] = "reg:linear"
params['eval_metric'] = 'mae'
params['eta'] = 0.01
#params['eta'] = 0.14693925
params['gamma'] = 0.7359492
params['min_child_weight'] = 6.3736338
params['colsample_bytree'] = 0.5995254
params['subsample'] = 0.6310414
params['max_depth'] = 8
params['max_delta_step'] = 0.086031
params['silent'] = 1
params['random_state'] = 929
​
d_train = xgb.DMatrix(X_train, label=y_train)
d_valid = xgb.DMatrix(X_valid, label=y_valid)
watchlist = [(d_train, 'train'), (d_valid, 'eval')]
    
    ## Build a model
clf = xgb.train(params,
                    d_train,
                    100000,
                    watchlist,
                    early_stopping_rounds=early_stopping)
​
    ## Evaluate model and predict
xgb_rounds.append(clf.best_iteration)
scores_val = clf.predict(d_valid, ntree_limit=clf.best_ntree_limit)
cv_score = mean_absolute_error(np.exp(y_valid), np.exp(scores_val))
print(' eval-MAE: %.6f' % cv_score)
y_pred_hot_opt = np.exp(clf.predict(d_test, ntree_limit=clf.best_ntree_limit))

## With chosen params - eval-MAE: 1150
## with lr of 0.01 -1147.516

solution = pd.DataFrame({"id":test_loader.id, "loss":y_pred_hot_opt})
solution[:20]
solution.to_csv("/Users/priyankadwivedi/Documents/Projects/Kaggle All State Insurance/sub1_oct26.csv", index = False)


## Look at important parameters
feat_imp = pd.Series(clf.get_fscore()).sort_values(ascending=False)[:20]
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')

fig, ax = plt.subplots()
#x_ax = np.exp(y_valid)
#y_ax = np.exp(scores_val)
x_ax = y_valid
y_ax = scores_val
ax.scatter(x_ax, y_ax, c='red')
ax.plot([-5,-1], [-5,-1], 'r-', lw=2)
ax.set_ylim([0,15])
ax.set_xlim([0,15])
ax.set_xlabel('Actual value')
ax.set_ylabel('Predicted value')
plt.grid()
plt.show()

print(len(y_pred_full))
# Print to csv
solution = pd.DataFrame({"id":test_loader.id, "loss":y_pred_full})
solution[:20]
solution.to_csv("/Users/priyankadwivedi/Documents/Projects/Kaggle All State Insurance/sub3_oct24.csv", index = False)

## LB score: 1127.57669

### XGB regressor on 5 folds using parameters from Bayesian Optimization 
# 1133 last night
d_test = xgb.DMatrix(test)
folds = 5
cv_sum = 0
early_stopping = 25
fpred = []
xgb_rounds = []
pred =[]
kf = KFold(features.shape[0], n_folds=folds)
for i, (train_index, test_index) in enumerate(kf):
    print('\n Fold %d\n' % (i + 1))
    X_train, X_val = features.iloc[train_index], features.iloc[test_index]
    y_train, y_val = labels.iloc[train_index], labels.iloc[test_index]
    ## Define cross validation variables 
    params = {}
    params['booster'] = 'gbtree'
    params['objective'] = "reg:linear"
    params['eval_metric'] = 'mae'
    params['eta'] = 0.1228
    params['gamma'] = 0.5668
    params['min_child_weight'] = 2.534
    params['colsample_bytree'] = 0.4769
    params['subsample'] = 0.5445
    params['max_depth'] = 6
    params['max_delta_step'] = 0.0836
    params['silent'] = 1
    params['random_state'] = 276

    d_train = xgb.DMatrix(X_train, label=y_train)
    d_valid = xgb.DMatrix(X_val, label=y_val)
    watchlist = [(d_train, 'train'), (d_valid, 'eval')]
    
    ## Build a model
    clf = xgb.train(params,
                    d_train,
                    100000,
                    watchlist,
                    early_stopping_rounds=early_stopping)

    ## Evaluate model and predict
    xgb_rounds.append(clf.best_iteration)
    scores_val = clf.predict(d_valid, ntree_limit=clf.best_ntree_limit)
    cv_score = mean_absolute_error(np.exp(y_val), np.exp(scores_val))
    print(' eval-MAE: %.6f' % cv_score)
    y_pred = np.exp(clf.predict(d_test, ntree_limit=clf.best_ntree_limit))
    
    ### Add predictions and average them 

    if i > 0:
        fpred = pred + y_pred
    else:
        fpred = y_pred
    pred = fpred
    cv_sum = cv_sum + cv_score

mpred_full = pred / folds
score = cv_sum / folds
print('\n Average eval-MAE: %.6f' % score)
n_rounds = int(np.mean(xgb_rounds))
##1133 on 5 folds