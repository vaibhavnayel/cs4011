from sklearn.ensemble import ExtraTreesClassifier
from mlxtend.feature_extraction import PrincipalComponentAnalysis as PCA
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt

X = np.genfromtxt('../../contest_data/xtrain_linear_imputed.csv', delimiter=',')
y = np.genfromtxt('../../contest_data/train.csv', delimiter=',')[1:,-1]




pca = PCA(n_components=1000)
X_pca = pca.fit(X).transform(X)
et = ExtraTreesClassifier(n_estimators=1000, max_depth=None, random_state=0,verbose=0)
scores = cross_val_score(et, X_pca, y,scoring='f1_micro',cv=5,verbose=5)
print scores.mean()


et = ExtraTreesClassifier(n_estimators=300, max_depth=None, random_state=0,verbose=1)
scores = cross_val_score(et, X, y,scoring='f1_micro',cv=5,verbose=5)
print scores.mean()
'''
components=1000,estimators=1000 gives 32.6% f1
'''





pca = PCA(n_components=1000)
X_pca = pca.fit(xtrain).transform(xtrain)
et = ExtraTreesClassifier(n_estimators=1000, max_depth=None, random_state=0,verbose=0)
scores = cross_val_score(et, X_pca, ytrain,scoring='f1_micro',cv=5,verbose=5)
print scores.mean()