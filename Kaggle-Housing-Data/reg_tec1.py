# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 04:33:36 2016

@author: priyankadwivedi
"""

## Best Kaggle RMSE: 0.163
## Polynomial Reg and SVC completely fail
# Look into nbr_cat - can only see worst 

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

paths = "/Users/priyankadwivedi/Documents/Projects/Kaggle Housing dataset"
picfile = os.path.join(paths, "df.pkl")

df = pd.read_pickle(picfile)
df.shape
df.head()

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
    #print("R2 (on data): {:.2}".format(r2)

# Define x and y
y = df['SalePrice']
y.shape
x = df.copy()
x.shape
x.drop(['SalePrice'], axis = 1, inplace = True)
x.shape
    
#TODO:1 Ordinary Linear Regression
from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
y = y
x = x
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
#0.81
kaggle_rmse(y,p)
#0.169   
# Store in a list
pred_linear = p   

#TODO:2 Ridge Regression CV
from sklearn import linear_model
met = linear_model.LassoCV(alphas=[0.1,0.5, 1.0, 10.0, 15.0, 20.0, 50.0, 100.0, 200.0])
#met = linear_model.RidgeCV(alphas=[0.1,0.5, 1.0, 10.0, 15.0, 20.0, 50.0])
y = y
x = x

kf = KFold(len(x), n_folds=5)
pred = np.zeros_like(y)
for train_index,test_index in kf:
        met.fit(x.iloc[train_index], y.iloc[train_index])
        pred[test_index] = met.predict(x.iloc[test_index])

print(y.min(), y.max(), y.mean())
print(pred.min(), pred.max(), pred.mean())

met.alpha_
print('RMSLE on 5 folds:', kaggle_rmse(y,pred))
# 0.1708 #0.163 with LassoCV
print('R2_score on 5 folds:', r2_score(y,pred))
#0.859
pred_lasso = pred

## See coefficients
pred_coef = met.coef_
for s in pred_coef:
    print(s)
# Print col_names
for col in x.columns:
    print(col)

#Print mean of each column
for col in x.columns:
    print x[col].mean()

met.intercept_
# Most powerful vars: YearBuilt, YearRemodAdd, 1stFlrSF, GrLivArea,OverallQual, 2ndFlrSF,TotRmsAbvGrd, TotalBsmtSF, BedroomAbvGr


#TODO:2 ElasticNet Regression
from sklearn.linear_model import ElasticNetCV
met = ElasticNetCV(l1_ratio=[.01, .05, .25, .5, .75, .95, .99])
y = y
x = x

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

# Store in a list
pred_elastic = pred

#Visualizing the outcome
fig, ax = plt.subplots()
y = y
li =ax.scatter(y, pred_linear, c='r')
en = ax.scatter(y, pred_elastic, c='b')
ax.plot([-5,-1], [-5,-1], 'r-', lw=2)
ax.set_xlabel('Actual value')
ax.set_ylabel('Predicted value')
ax.set_ylim([0,800000])
ax.set_xlim([0,800000])
plt.legend((li, en),
           ('Pred_Linear', 'Pred_Elastic'),
           scatterpoints=1,
           loc='best',
           ncol=3,
           fontsize=8)
fig.savefig('Results of Regressions')   

#TODO: 3- Decision Tree regressor
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeRegressor
lr = DecisionTreeRegressor(random_state = 4)
y = y
x = x
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
#0.73
kaggle_rmse(y,p)
#0.211

#Optimizing the DT regression
max_depth = (2,3,4,5,6,7,8,9,10)
for depth in max_depth:
    lr = DecisionTreeRegressor(random_state = 4, max_depth = depth)
    y = y
    x = x
    kf = KFold(len(x), n_folds=5, shuffle=True, random_state=4)
    p = np.zeros_like(y)
    for train_index,test_index in kf:
        lr.fit(x.iloc[train_index], y.iloc[train_index])
        p[test_index] = lr.predict(x.iloc[test_index])
    kaggle_rmse(y,p)
    
#Optimal depth: 6 - Run for optimal depth
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeRegressor
lr = DecisionTreeRegressor(random_state = 4, max_depth =6)
y = y
x = x
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
#0.77
kaggle_rmse(y,p)
#0.193

pred_dt = p

#Draw scatter plot
#Visualizing the outcome
fig, ax = plt.subplots()
y = y
li =ax.scatter(y, pred_linear, c='r')
en = ax.scatter(y, pred_elastic, c='b')
dt = ax.scatter(y, pred_dt, c='g')
ax.plot([-5,-1], [-5,-1], 'r-', lw=2)
ax.set_xlabel('Actual value')
ax.set_ylabel('Predicted value')
ax.set_ylim([0,800000])
ax.set_xlim([0,800000])
plt.legend((li, en, dt),
           ('Pred_Linear', 'Pred_Elastic', 'Pred Dt'),
           scatterpoints=1,
           loc='best',
           ncol=3,
           fontsize=8)
fig.savefig('Results of Regressions')   
"""
#TODO 4: Support Vector Regression 
from sklearn.metrics import make_scorer
scoring_fnc = make_scorer(kaggle_rmse)
from sklearn.cross_validation import KFold
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
met = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                               "gamma": np.logspace(-2, 2, 5)}, scoring = scoring_fnc )
y = y
x = x
kf = KFold(len(x), n_folds=5)
pred = np.zeros_like(y)
for train_index,test_index in kf:
        met.fit(x.iloc[train_index], y.iloc[train_index])
        pred[test_index] = met.predict(x.iloc[test_index])

print(y.min(), y.max(), y.mean())
print(pred.min(), pred.max(), pred.mean())

print(met.best_estimator_)

print('RMSLE on 5 folds:', kaggle_rmse(y,pred))
#0.400
r2_score(y,pred)
#-0.51


#TODO 5: Polynomial regression
from sklearn.metrics import make_scorer
scoring_fnc = make_scorer(kaggle_rmse)
from sklearn.cross_validation import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from sklearn.grid_search import GridSearchCV
met = Pipeline([('poly', PolynomialFeatures(degree=2)),
                 ('linear', LinearRegression(fit_intercept=False))])
y = y
x = x
kf = KFold(len(x), n_folds=5)
pred = np.zeros_like(y)
for train_index,test_index in kf:
        met.fit(x.iloc[train_index], y.iloc[train_index])
        pred[test_index] = met.predict(x.iloc[test_index])

print(y.min(), y.max(), y.mean())
print(pred.min(), pred.max(), pred.mean())

print(met.best_estimator_)

print('RMSLE on 5 folds:', kaggle_rmse(y,pred))
#0.400
r2_score(y,pred)
#-0.51
"""