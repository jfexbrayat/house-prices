"""
27/07/2018 - JFE
this script predicts house prices using a decision implemented in scikit
using out of box features
"""

#first import useful modules
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import GridSearchCV, train_test_split

def msle(true,pred):
    return np.mean((np.log(true)-np.log(pred))**2)

#get training data and extract target variables
dftrain = pd.read_csv('train.csv',index_col=0)
y = dftrain.SalePrice
dftrain =  dftrain.drop('SalePrice',axis=1)
#load test data
dftest = pd.read_csv('test.csv',index_col=0)
#concatenate for categories definition
df = pd.concat([dftrain,dftest])
#get only columns with numbers
numcols = df.dtypes!='O'
#fillna
df.loc[:,numcols] = df.loc[:,numcols].fillna(0)
df.loc[:,~numcols]= df.loc[:,~numcols].fillna('dummy')
#get predictors
X = pd.get_dummies(df,drop_first=True).iloc[:dftrain.shape[0]]
X_pred = pd.get_dummies(df,drop_first=True).iloc[-dftest.shape[0]:]

#create train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state = 42)

#define parameters
param_grid = {'colsample_bytree': [0.3, 0.7],
    'n_estimators': [50,100,500],
    'max_depth': [2, 5],
    'learning_rate': [0.001,0.01,0.1]}

gbm = xgb.XGBRegressor(n_jobs = -1)
grid = GridSearchCV(gbm,param_grid = param_grid,verbose = True,scoring='neg_mean_squared_log_error',cv=4)
grid.fit(X_train,y_train)

print(grid.best_params_)

model = grid.best_estimator_

print(msle(y_train,model.predict(X_train))**(1/2))
print(msle(y_test,model.predict(X_test))**(1/2))

df_pred = pd.DataFrame(model.predict(X_pred),index=X_pred.index,columns=['SalePrice'])
