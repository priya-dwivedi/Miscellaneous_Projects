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
import xgboost as xgb


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


# factorize categorical features
for feat in cats:
   train_test[feat] = pd.factorize(train_test[feat], sort=True)[0]

train_test['cat100'][:10]


# Split back into test and train
x_train = train_test.iloc[:ntrain, :]
x_test = train_test.iloc[ntrain:, :]


train = x_train.copy()
test = x_test.copy()
train.shape
train.head()


## Transform target into np log of loss
train_labels = np.log(np.array(train_loader['loss']))
train_ids = train_loader['id'].values.astype(np.int32)
test_ids = test_loader['id'].values.astype(np.int32)


## Lets now look at only scorecard 1
## Break training into cont14<=0.5 and cont14> 0.5
train2 = train.copy()
train2['loss'] = train_labels


## Lets do xgboost on train_less. 
labels = train2['loss']
labels.head()

features = train2.copy()
features.drop(['loss'], axis = 1, inplace = True)
features.head()

## Imports for XGboost tuning
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

## Tune using Bayesian Optimization
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
xtrain = features
y = labels
n_folds = 5
i = 0
### Bayesian optimization on the train_less dataset
from sklearn.cross_validation import cross_val_score, cross_val_predict
folds = 5

kf = KFold(len(labels), n_folds=n_folds, shuffle=True)
for inTr, inTe in kf:
    ## Create datasets
    xtr = xtrain.iloc[inTr]
    ytr = y.iloc[inTr]
    xte = xtrain.iloc[inTe]
    yte = y.iloc[inTe]

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

#max_depth	6.557666667
#lr	0.122866667
#gamma	0.566866667
#min_child_weight	2.534
#subsample	0.544566667
#colsample	0.4769
#max_delta	0.083603333


## Lets reshuffle and run on full dataset
## Approach - For scorecard less than 0,5 (scorecard 1), we will tune using xgb
from sklearn.cross_validation import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(features, labels, test_size=0.25, random_state=1981)

print("Training Data", X_train.shape, y_train.shape)
print("Validation Data", X_valid.shape, y_valid.shape)

## Define cross validation variables 
d_test = xgb.DMatrix(test)
early_stopping = 25
xgb_rounds = []

params = {}
params['booster'] = 'gbtree'
params['objective'] = "reg:linear"
params['eval_metric'] = 'mae'
params['eta'] = 0.01
#params['eta'] = 0.1228
params['gamma'] = 0.5668
params['min_child_weight'] = 2.534
params['colsample_bytree'] = 0.4769
params['subsample'] = 0.5445
params['max_depth'] = 6
params['max_delta_step'] = 0.0836
params['silent'] = 1
params['random_state'] = 929

d_train = xgb.DMatrix(X_train, label=y_train)
d_valid = xgb.DMatrix(X_valid, label=y_valid)
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
cv_score = mean_absolute_error(np.exp(y_valid), np.exp(scores_val))
print(' eval-MAE: %.6f' % cv_score)
y_pred_full = np.exp(clf.predict(d_test, ntree_limit=clf.best_ntree_limit))

## With chosen params - eval-MAE: 1153.009301
## No of runs is 14670
## with lr set to 0.01 - 1147.68

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
    params['eta'] = 0.01
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
##1147 on full set - LB score: 1127.57669
## Lets see what happens with cross folds and the impact on LB score