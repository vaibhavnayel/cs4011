# Importing Relevant Libraries

import pandas as pd 
import numpy as np
import math  
from sklearn import tree 

# Impas_matorting Datasets

X_train = pd.read_csv('../contest_data/train.csv')
X_test = pd.read_csv('../contest_data/test.csv')

# Slicing and Dicing the Dataset

X_train_pred  = X_train.iloc[:,501:2601]
X_train_missing = X_train.iloc[:,1:501]

#Output the Data 

impute_add = list()
for column in X_train_missing: 
	print "Column: " + column
	impute_X = pd.concat([X_train_missing[column],X_train_pred],axis=1,join='inner')
	impute_X_train = impute_X.dropna(axis=0, how='any')       
	impute_y_train = impute_X_train.iloc[:,0:1]
	impute_X_train = impute_X_train.drop([column], axis=1)
	impute_X_predict = impute_X[impute_X[column].isin([np.nan])]
	impute_X_predict = impute_X_predict.drop([column],axis=1)
	clf = tree.DecisionTreeRegressor()
	clf = clf.fit(impute_X_train, impute_y_train)
	impute_y_predict = clf.predict(impute_X_predict)
	impute_add.append(impute_y_predict)

X_train_matrix = X_train.as_matrix()
X_train_matrix = X_train_matrix.T[1:]
for i in range(len(X_train_matrix)):
     index=0
     print i
     for j in range(len(X_train_matrix[i])):
     	if np.isnan(X_train_matrix[i][j]) == True:
        	X_train_matrix[i][j] = impute_add[i][index]
        	index = index+1

X_train_matrix = X_train_matrix.T
X_train_matrix = pd.DataFrame(X_train_matrix)
X_train_matrix.to_csv('../contest_data/train_desicion_tree_impute.csv')