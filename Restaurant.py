# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 16:42:53 2020

@author: Hp
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.pandas.set_option('display.max_columns',None)

dataset = pd.read_csv('train.csv')
dataset_test = pd.read_csv('test.csv')

dataset.head()

features_with_na = [features for features in dataset.columns if dataset[features].isnull().sum()>1]#no na values 
features_with_na_test = [features for features in dataset_test.columns if dataset_test[features].isnull().sum()>1]#no na values 

numerical_features = [features for features in dataset.columns if dataset[features].dtypes!='O']
print('Number of Numerical features:', len(numerical_features))

dataset[numerical_features]

dataset['Open Date'] = pd.to_datetime(dataset['Open Date'], format='%m/%d/%Y')   
dataset_test['Open Date'] = pd.to_datetime(dataset_test['Open Date'], format='%m/%d/%Y')
dataset['OpenDays']=""
dataset_test['OpenDays']=""

currentdate = pd.DataFrame({'Date':np.repeat(['01/01/2015'],[len(dataset)]) })
currentdate_test = pd.DataFrame({'Date':np.repeat(['01/01/2015'],[len(dataset_test)]) })
currentdate['Date'] = pd.to_datetime(currentdate['Date'], format='%m/%d/%Y')  
currentdate_test['Date'] = pd.to_datetime(currentdate_test['Date'], format='%m/%d/%Y')  

dataset['OpenDays'] = currentdate['Date'] - dataset['Open Date']
dataset['OpenDays'] = dataset['OpenDays'].astype('timedelta64[D]').astype(int)

dataset_test['OpenDays'] = currentdate_test['Date'] - dataset_test['Open Date']
dataset_test['OpenDays'] = dataset_test['OpenDays'].astype('timedelta64[D]').astype(int)

dataset = dataset.drop(['Id','Open Date'],axis=1)
categorical_features=[feature for feature in dataset.columns if dataset[feature].dtypes=='O']
categorical_features

dataset_test = dataset_test.drop(['Id','Open Date'],axis=1)
categorical_features=[feature for feature in dataset.columns if dataset[feature].dtypes=='O']
categorical_features

for feature in categorical_features:
    data=dataset.copy()
    data.groupby(feature)['revenue'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('revenue')
    plt.title(feature)
    plt.show()
    

    
def tenure_lab(dataset) :
    
    if dataset["OpenDays"] <= 1000:
        return "0-1000"
    elif (dataset["OpenDays"] > 1000) & (dataset["OpenDays"]<= 2000 ):
        return "1000-2000"
    elif (dataset["OpenDays"] > 2000) & (dataset["OpenDays"] <= 3000) :
        return "2000-3000"
    elif (dataset["OpenDays"] > 3000) & (dataset["OpenDays"] <= 4000) :
        return "3000-4000"
    elif (dataset["OpenDays"] > 4000) & (dataset["OpenDays"] <= 5000) :
        return "3000-4000"
    elif dataset["OpenDays"] > 5000 :
        return ">5000"
dataset["OpenDays_group"] = dataset.apply(lambda dataset:tenure_lab(dataset),axis = 1)
dataset_test["OpenDays_group"] = dataset_test.apply(lambda dataset_test:tenure_lab(dataset_test),axis = 1)
dataset.groupby('OpenDays_group')['revenue'].median().plot.bar()
plt.xlabel(feature)
plt.ylabel('revenue')
plt.title(feature)
plt.show()

dataset  = dataset.drop("OpenDays",axis =1 )
dataset = dataset.drop(['City'],axis = 1)
dataset = pd.get_dummies(dataset,columns = ['City Group', 'Type', 'OpenDays_group'],drop_first=True)
dataset_test  = dataset_test.drop("OpenDays",axis =1 )
dataset_test = dataset_test.drop(['City'],axis = 1)
dataset_test = pd.get_dummies(dataset_test,columns = ['City Group', 'Type', 'OpenDays_group'],drop_first=True)
#correlation = dataset.corr()
#correlation.to_csv("corr.csv")

#dataset =  dataset.iloc[:,[0,1,3,5,6,10,13,14,16,18,19,20,21,22,23,24,27,37,39,42,43,44]]
#dataset



#Feature Selection
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
y_train=dataset[['revenue']]
x_train = dataset.drop(['revenue'],axis = 1)


feature_sel_model = SelectFromModel(Lasso(alpha=0.005, random_state=0))
feature_sel_model.fit(x_train, y_train)

feature_sel_model.get_support()
selected_feat = x_train.columns[(feature_sel_model.get_support())]

x_train=x_train[selected_feat]

X_features= x_train.columns

from statsmodels.stats.outliers_influence import variance_inflation_factor
def get_vif_factors( X ):
    X_matrix = X.values
    vif = [ variance_inflation_factor( X_matrix, i ) for i in range( X_matrix.shape[1] ) ]
    vif_factors = pd.DataFrame()s
    vif_factors['column'] = X.columns
    vif_factors['vif'] = vif
    return vif_factors

vif_factors = get_vif_factors( x_train[X_features] )
vif_factors

columns_with_lessvif = vif_factors[vif_factors.vif <= 7].column

x_train = x_train[columns_with_lessvif]
dataset_test = dataset_test[columns_with_lessvif]

# from sklearn.model_selection import cross_val_score
# # function to get cross validation scores
# def get_cv_scores(model):
#     scores = cross_val_score(model,
#                              x_train,
#                              y_train,
#                              cv=10,scoring='accuracy')
    
#     print('CV Mean: ',scores)
#     print('STD: ', np.std(scores))
#     print('\n')from sklearn.linear_model import LinearRegression
# # Train model
# lr = LinearRegression().fit(x_train, y_train)

# get_cv_scores(lr)

# from sklearn.linear_model import Ridge
# # Train model with default alpha=1
# ridge = Ridge(alpha=1).fit(x_train, y_train)
# # get cross val scores
# get_cv_scores(ridge)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import cross_val_score
    
regressors = {
    'Linear Regression' : LinearRegression(),
    'Decision Tree' : DecisionTreeRegressor(),
    'Random Forest' : RandomForestRegressor(),
    'Support Vector Machines' : SVR(),
}
results=pd.DataFrame(columns=['MAE','MSE','R2-score'])

for method,func in regressors.items():
    func.fit(x_train,y_train)
    pred = func.predict(x_test)
    results.loc[method]= [mean_absolute_error(y_test,pred),
                          mean_squared_error(y_test,pred),
                          r2_score(y_test,pred)
                         ]
results