from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from mlxtend.feature_extraction import PrincipalComponentAnalysis as PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV

X = np.genfromtxt('../../contest_data/xtrain_linear_imputed.csv', delimiter=',')
print 'loaded X'
y = np.genfromtxt('../../contest_data/train.csv', delimiter=',')[1:,-1]
print 'loaded y'

svc=SVC(C=1,gamma='auto',verbose=1)
bag=BaggingClassifier(base_estimator=svc, n_estimators=10, max_samples=1.0, 
	max_features=0.2, bootstrap=True, bootstrap_features=False, oob_score=False, 
	warm_start=False, n_jobs=1, random_state=0, verbose=1)
scores = cross_val_score(bag, X, y,scoring='f1_micro',cv=5,verbose=5)
print scores.mean()