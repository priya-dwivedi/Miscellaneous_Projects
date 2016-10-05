# -*- coding: utf-8 -*-
"""
Created on Mon Oct 03 10:43:56 2016

@author: s6324900
"""

import numpy as np
import pandas as pd

import os
paths = "C:\Users\s6324900\Desktop\housing dataset"
filename = os.path.join(paths, "train.csv")

df =  pd.DataFrame.from_csv(filename)
df.head()
df.shape

from matplotlib import pyplot as plt
plt.scatter(df['LotArea'], df['SalePrice'], color='r')
plt.scatter(df['GarageArea'], df['SalePrice'], color='r')

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
x = df['GarageArea']
y = df['SalePrice']
x = np.transpose(np.atleast_2d(x))
lr.fit(x, y)
y_predicted = lr.predict(x)

plt.scatter(df['GarageArea'], df['SalePrice'], color='r')
plt.plot(df['GarageArea'], y_predicted, color = 'b')
plt.show()

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
    
kaggle_rmse(y, y_predicted)
#RMSE (of training data): 0.316

# Compute R2 score
def r2_score(y,p):
    from sklearn.metrics import r2_score
    r2 = r2_score(y, p)
    print("R2 (on data): {:.2}".format(r2))
r2_score(y, y_predicted)
#0.39

df.dtypes
# Create a new dataframe with all non-object columns
col_list = []
for col in df.columns:
        if (df[col].dtypes != 'object'): # non-object columns
            col_list.append(col)

print(col_list)

df2 = df.loc[:,col_list]
df2.head()
df2.shape

df2.dtypes
# Plot using all available vars
df3 = df2.fillna(0)
y = df3['SalePrice']
x = df3.drop('SalePrice', axis =1)
lr.fit(x,y)
p = lr.predict(x)

print(y.min(), y.max())
print(p.min(), p.max())
print(len(y), len(p))

# Draw plot
plt.scatter(p, y)
plt.xlabel('Predicted price')
plt.ylabel('Actual price')
plt.plot([y.min(), y.max()], [[y.min()], [y.max()]])

#Calculate Kaggle RMSLE
kaggle_rmse(y,p) # 0.333
r2_score(y,p)
#0.82


# Cross Validation folds
from sklearn.cross_validation import KFold
y = df3['SalePrice']
x = df3.drop('SalePrice', axis =1)
kf = KFold(len(x), n_folds=5, indices=True, shuffle=True, random_state=4)
p = np.zeros_like(y)
for train_index,test_index in kf:
    lr.fit(x.iloc[train_index], y.iloc[train_index])
    p[test_index] = lr.predict(x.iloc[test_index])
    
r2_score(y,p)
#0.76
kaggle_rmse(y,p)
#0.34

##Penalized Regression
# L1 regularization: Penalize based on error - Called Lasso Regression
# L2 regularization: Penalize based on square of error - Ridge Regression
# Using both is called ElasticNet model

from sklearn.linear_model import ElasticNet, Lasso
en = ElasticNet(alpha=0.5)
ls = Lasso(alpha = 0.5)

# Cross Validation folds on Elastic Regression
from sklearn.cross_validation import KFold
y = df3['SalePrice']
x = df3.drop('SalePrice', axis =1)
kf = KFold(len(x), n_folds=5, indices=True, shuffle=True, random_state=4)
p = np.zeros_like(y)
for train_index,test_index in kf:
    ls.fit(x.iloc[train_index], y.iloc[train_index])
    p[test_index] = ls.predict(x.iloc[test_index])

r2_score(y,p)
#0.76
kaggle_rmse(y,p)
#0.333

## Do in one go
y = df3['SalePrice']
x = df3.drop('SalePrice', axis =1)
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge

for name, met in [
        ('linear regression', LinearRegression()),
        ('lasso()', Lasso()),
        ('elastic-net(.5)', ElasticNet(alpha=0.5)),
        ('lasso(.5)', Lasso(alpha=0.5)),
        ('ridge(.5)', Ridge(alpha=0.5)),
]:
    # Fit on the whole data:
    met.fit(x, y)

    # Predict on the whole data:
    p = met.predict(x)
    r2_train = r2_score(y, p)

    # Now, we use 10 fold cross-validation to estimate generalization error
    kf = KFold(len(x), n_folds=5)
    p = np.zeros_like(y)
    for train_index,test_index in kf:
        met.fit(x.iloc[train_index], y.iloc[train_index])
        p[test_index] = met.predict(x.iloc[test_index])

    r2_cv = r2_score(y, p)
    print('Method: {}'.format(name))
    print('R2 on training: {}'.format(r2_train))
    print('R2 on 5-fold CV: {}'.format(r2_cv))
    print()
    print()

## Visualize the Lasso path
# Convert x dataframe into a numpy array
y = df3['SalePrice']
x = df3.drop('SalePrice', axis =1)
x = x.as_matrix()
#x /= x.std(axis=0) 
from sklearn import linear_model
print("Computing regularization path using the LARS ...")
alphas, _, coefs = linear_model.lars_path(x, y, method='lasso', verbose=True)

xx = np.sum(np.abs(coefs.T), axis=1)
xx /= xx[-1]

plt.plot(xx, coefs.T)
ymin, ymax = plt.ylim()
plt.vlines(xx, ymin, ymax, linestyle='dashed')
plt.xlabel('|coef| / max|coef|')
plt.ylabel('Coefficients')
plt.title('LASSO Path')
plt.axis('tight')
plt.show()

## How to interpret
# When coef/max_coef = 1, then we are at unpenalized regression and coefficients are
# close to their value for unpenalized regression
# When coef/max_coef = 0 then most coefficients are set to zero 


# Construct an ElasticNetCV object - Almost always the best choice  
## It will test models that are almost like Ridge (when l1_ratio is 0.01 or 0.05) 
# as well as models that are almost like Lasso (when l1_ratio is 0.95 or 0.99)

from sklearn.linear_model import ElasticNetCV
met = ElasticNetCV(l1_ratio=[.01, .05, .25, .5, .75, .95, .99])
y = df3['SalePrice']
x = df3.drop('SalePrice', axis =1)

kf = KFold(len(x), n_folds=5)
pred = np.zeros_like(y)
for train_index,test_index in kf:
        met.fit(x.iloc[train_index], y.iloc[train_index])
        pred[test_index] = met.predict(x.iloc[test_index])


print('RMSLE on 5 folds:', kaggle_rmse(y,pred))

print('R2_score on 5 folds:', r2_score(y,pred))

print('')


fig, ax = plt.subplots()
y = y
ax.scatter(y, pred, c='k')
ax.plot([-5,-1], [-5,-1], 'r-', lw=2)
ax.set_xlabel('Actual value')
ax.set_ylabel('Predicted value')
fig.savefig('Figure_10k_scatter_EN_l1_ratio.png')

## Do Lasso regression to figure out which x's are important 
from sklearn.linear_model import ElasticNet, Lasso
ls = Lasso(alpha = 0.7)

# Cross Validation folds on Elastic Regression
from sklearn.cross_validation import KFold
y = df3['SalePrice']
x = df3.drop('SalePrice', axis =1)
kf = KFold(len(x), n_folds=5, indices=True, shuffle=True, random_state=4)
p = np.zeros_like(y)
for train_index,test_index in kf:
    ls.fit(x.iloc[train_index], y.iloc[train_index])
    p[test_index] = ls.predict(x.iloc[test_index])

r2_score(y,p)
#0.76
kaggle_rmse(y,p)
#0.333

x.shape
print(ls.coef_).reshape(36, 1).astype(float)
for col in x.columns:
    print x[col].mean()
