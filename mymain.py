#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import statsmodels.api as sm
import patsy
import warnings
warnings.filterwarnings("ignore")


# In[2]:


def preprocess(data):
    tmp = pd.to_datetime(data['Date'])
    data['Wk'] = tmp.dt.isocalendar().week
    data['Yr'] = tmp.dt.year
    data['Wk'] = pd.Categorical(data['Wk'], categories=[i for i in range(1, 53)]) 
    return data

def svd_smoothing(X, d):
    
    numeric_columns = X.select_dtypes(include=np.number).columns
    X_mean = X[numeric_columns].mean(axis=0).values 
    X_centered = X[numeric_columns].sub(X_mean, axis=1)

    U, D, Vt = np.linalg.svd(X_centered, full_matrices=False)

    U_d = U[:, :d]
    D_d = np.diag(D[:d])
    Vt_d = Vt[:d, :]

    Xbar_values = np.dot(U_d, np.dot(D_d, Vt_d)).T + X_mean[:, np.newaxis]
    
    Xbar = pd.DataFrame(Xbar_values)

    return Xbar

def shift(data, percentage):
    for index, row in data.iterrows():
        if row['Wk'] == 51:
            reduction_amount = row['Weekly_Pred'] * percentage
            data.at[index, 'Weekly_Pred'] -= reduction_amount
            next_week_index = index + 1
            if next_week_index < len(data) and data.at[next_week_index, 'Wk'] == 52:
                data.at[next_week_index, 'Weekly_Pred'] += reduction_amount

    return data


# In[3]:


d_components = 8

train = pd.read_csv("train.csv")
train2 = train.copy()
train2['Weekly_Sales'] = train2['Weekly_Sales'].replace(0, 0.1)

for p in train.Dept.unique():

        
    filtered_train = train2[train2['Dept'] == p]

    selected_columns = filtered_train[['Store', 'Date', 'Weekly_Sales']]

    train_dept_ts = selected_columns.pivot(index='Date', columns='Store', values='Weekly_Sales').reset_index()
    
    train_dept_ts.fillna(0, inplace=True)


    X_train_smoothed = svd_smoothing(train_dept_ts, d_components)

    melt = pd.melt(X_train_smoothed)
    values = melt['value']

    train_dept_ts2 = train_dept_ts.copy()
    shape = train_dept_ts2.shape
    
    values = np.array(values).reshape(shape[0], shape[1]-1)
    zero_indices = np.where(train_dept_ts2 == 0.0)
    zero_coordinates = list(zip(zero_indices[0], zero_indices[1]-1))
    zero_coordinates
    for coordinate in zero_coordinates:
        values[coordinate] = np.nan

        
    values = pd.DataFrame(values)
    values = values.T

    values = np.array(values).reshape(shape[0] * (shape[1]-1),)
    values = pd.DataFrame(values)

    values = values.dropna()
    values.index = train2[train2['Dept'] == p].index
    train2.loc[train2['Dept'] == p, 'Weekly_Sales'] = values[0]


test = pd.read_csv("test.csv")
test_pred = pd.DataFrame()

train_pairs = train2[['Store', 'Dept']].drop_duplicates(ignore_index=True)
test_pairs = test[['Store', 'Dept']].drop_duplicates(ignore_index=True)
unique_pairs = pd.merge(train_pairs, test_pairs, how = 'inner', on =['Store', 'Dept'])

train_split = unique_pairs.merge(train2, on=['Store', 'Dept'], how='left')
train_split = preprocess(train_split)

    
y, X = patsy.dmatrices('Weekly_Sales ~ Weekly_Sales + Store + Dept + Yr + np.power(Yr, 2) + Wk', 
                        data = train_split, 
                        return_type='dataframe')
train_split = dict(tuple(X.groupby(['Store', 'Dept'])))


test_split = unique_pairs.merge(test, on=['Store', 'Dept'], how='left')
test_split = preprocess(test_split)

    
y, X = patsy.dmatrices('Yr ~ Store + Dept + Yr + np.power(Yr, 2) + Wk', data=test_split, return_type='dataframe')
X['Date'] = test_split['Date']
test_split = dict(tuple(X.groupby(['Store', 'Dept'])))

keys = list(train_split)



    

for key in keys:
    X_train = train_split[key]
    X_test = test_split[key]
 
    Y = X_train['Weekly_Sales']
    X_train = X_train.drop(['Weekly_Sales','Store', 'Dept'], axis=1)
    
    cols_to_drop = X_train.columns[(X_train == 0).all()]
    X_train = X_train.drop(columns=cols_to_drop)
    X_test = X_test.drop(columns=cols_to_drop)
 
    cols_to_drop = []
    for i in range(len(X_train.columns) - 1, 1, -1):
        col_name = X_train.columns[i]
        tmp_Y = X_train.iloc[:, i].values
        tmp_X = X_train.iloc[:, :i].values

        coefficients, residuals, rank, s = np.linalg.lstsq(tmp_X, tmp_Y, rcond=None)
        if np.sum(residuals) < 1e-10:
                cols_to_drop.append(col_name)
        
    X_train = X_train.drop(columns=cols_to_drop)
        
    X_test = X_test.drop(columns=cols_to_drop)

    model = sm.OLS(Y, X_train).fit()
    mycoef = model.params.fillna(0)
    
    tmp_pred = X_test[['Store', 'Dept', 'Date']]
    X_test = X_test.drop(['Store', 'Dept', 'Date'], axis=1)
    
    tmp_pred['Weekly_Pred'] = np.dot(X_test, mycoef)

    # adding isholiday
    tmp_pred = tmp_pred.merge(test[['Store', 'Dept', 'Date', 'IsHoliday']], on=['Store', 'Dept', 'Date'], how='left')

    test_pred = pd.concat([test_pred, tmp_pred], ignore_index=True)
    test_pred = preprocess(test_pred)
        

if test_pred.Date[0] == '2011-11-04':
    test_pred = shift(test_pred, 1/14)
        
test_pred['Weekly_Pred'].fillna(0, inplace=True)
test_pred = test_pred[['Store', 'Dept', 'Date', 'IsHoliday', 'Weekly_Pred']]
test_pred.to_csv("mypred.csv", index=False)



