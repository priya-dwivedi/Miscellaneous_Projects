# -*- coding: utf-8 -*-
"""
Created on Fri Oct 07 13:40:00 2016

@author: s6324900
"""
## Best Kaggle RMSE: 0.156
## Polynomial Reg and SVC completely fail
# Look into nbr_cat - can only see worst 

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

paths = "C:\Users\s6324900\Desktop\housing dataset"
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
    
def kaggle_rmse_log(y,p):
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y,p)
    rmse = np.sqrt(mse)
    return rmse


# Compute R2 score
def r2_score(y,p):
    from sklearn.metrics import r2_score
    r2 = r2_score(y, p)
    return r2
    #print("R2 (on data): {:.2}".format(r2))


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
# y=y
y = df['SalePrice']
y = np.log(y)
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
#0.827
kaggle_rmse(y,p)
#0.1607  

kaggle_rmse_log(y,p)
#0.145
 
# Store in a list
pred_linear = p   

#TODO:2  Laso Regression and Ridge Regression CV
from sklearn import linear_model
#met = linear_model.LassoCV(alphas=[0.1,0.5, 1.0, 10.0, 15.0, 20.0, 50.0, 100.0, 200.0])
met = linear_model.RidgeCV(alphas=[0.1,0.5, 1.0, 10.0, 15.0, 20.0, 50.0])
#y = y
y = df['SalePrice']
y = np.log(y)
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
#0.156 with LassoCV
#0.159 with Ridge Regression
print('R2_score on 5 folds:', r2_score(y,pred))
#0.869 with Lasso and Ridge

kaggle_rmse_log(y,pred)
#0.172 - Lasso
#0.131 - Ridge Regression; alpha = 15.0


## Make predictions on test dataste
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

pred_lasso = pred

pred_ridge = pred

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
# Most powerful vars: YearBuilt, YearRemodAdd, 1stFlrSF, GrLivArea, OverallQual, 2ndFlrSF,TotRmsAbvGrd, TotalBsmtSF, BedroomAbvGr


#TODO:2 ElasticNet Regression
from sklearn.linear_model import ElasticNetCV
met = ElasticNetCV(l1_ratio=[.01, .05, .25, .5, .75, .95, .99])
y = df['SalePrice']
y = np.log(y)
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
#0.801

kaggle_rmse_log(y,pred)
#0.178

# Store in a list
pred_elastic = pred

#Visualizing the outcome
fig, ax = plt.subplots()
y = y
li =ax.scatter(y, pred_linear, c='r')
lo =ax.scatter(y, pred_lasso, c='b')
ri =ax.scatter(y, pred_ridge, c='g')
#en = ax.scatter(y, pred_elastic, c='b')
ax.plot([-5,-1], [-5,-1], 'r-', lw=2)
ax.set_xlabel('Actual value')
ax.set_ylabel('Predicted value')
ax.set_ylim([10,15])
ax.set_xlim([10,15])
plt.legend((li, lo, ri),
           ('Pred_Linear', 'Pred Lasso', 'Pred Ridge'),
           scatterpoints=1,
           loc='best',
           ncol=3,
           fontsize=8)
fig.savefig('Results of Regressions')   

#TODO: 3- Decision Tree regressor with GridSearch 
from sklearn.metrics import make_scorer
scoring_fnc = make_scorer(kaggle_rmse)
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.grid_search import GridSearchCV
met = GridSearchCV(DecisionTreeRegressor(random_state = 27), cv=5,
                   param_grid={"max_depth": [2,3,4,5,6,7,8,9,10]
                              }, scoring = scoring_fnc )

y = df['SalePrice']
y = np.log(y)
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
# 0.272
print('R2_score on 5 folds:', r2_score(y,pred))
#0.582

kaggle_rmse_log(y,pred)
#0.253

pred_dt = pred

#Visualizing the outcome
fig, ax = plt.subplots()
y = y
li =ax.scatter(y, pred_linear, c='r')
lo = ax.scatter(y, pred_lasso, c='b')
dt = ax.scatter(y, pred_dt, c='g')

#en = ax.scatter(y, pred_elastic, c='b')
ax.plot([-5,-1], [-5,-1], 'r-', lw=2)
ax.set_xlabel('Actual value')
ax.set_ylabel('Predicted value')
ax.set_ylim([0,800000])
ax.set_xlim([0,800000])
plt.legend((li, lo, dt),
           ('Pred Linear', 'Pred_Lasso', 'Pred DT'),
           scatterpoints=1,
           loc='best',
           ncol=3,
           fontsize=8)
fig.savefig('Results of Regressions')   


## Best results so far
from sklearn import linear_model
met = linear_model.RidgeCV(alphas=[0.1,0.5, 1.0, 10.0, 15.0, 20.0, 50.0])

y = df['SalePrice']
y = np.log(y)
x = x

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

kaggle_rmse_log(y,pred)


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

pred_sale = np.exp(pred)
pred_sale.shape


fig, ax = plt.subplots()
y = df['SalePrice']
li =ax.scatter(y, pred_sale, c='r')
#lo = ax.scatter(y, pred_lasso, c='b')
#dt = ax.scatter(y, pred_dt, c='g')

#en = ax.scatter(y, pred_elastic, c='b')
ax.plot([-5,-1], [-5,-1], 'r-', lw=2)
ax.set_xlabel('Actual value')
ax.set_ylabel('Predicted value')
ax.set_ylim([0,800000])
ax.set_xlim([0,800000])
fig.savefig('Results of Regressions')   


## Neural Network Regression
## Gradient Boosting Machine 

#TODO: 3- Decision Tree regressor with GridSearch 
from sklearn.metrics import make_scorer
scoring_fnc = make_scorer(kaggle_rmse)
from sklearn.cross_validation import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.grid_search import GridSearchCV
met = GridSearchCV(MLPRegressor(random_state = 27), cv=5,
                   param_grid={"activation": ['logistic', 'relu'],
                               "solver" : ['sgd', 'adam'],
                                "learning_rate_init" : [0.001, 0.05, 0.01, 0.5],
                              }, scoring = scoring_fnc )
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
# 0.272
print('R2_score on 5 folds:', r2_score(y,pred))
#0.582

