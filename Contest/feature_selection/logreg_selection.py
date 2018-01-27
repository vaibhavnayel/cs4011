import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn import linear_model


#load data
xtrain_imputed = np.genfromtxt('../../contest_data/xtrain_density_imputed.csv', delimiter=',')[:,500:]
ytrain = np.genfromtxt('../../contest_data/train.csv', delimiter=',')[1:,-1]
ytrain=np.asmatrix(ytrain).T

#logistic regression for recursive feature selection
logreg = linear_model.LogisticRegression(C=1.0)
rfecv = RFECV(estimator=logreg, step=200, cv=StratifiedKFold(2),
              scoring='accuracy',verbose=10)
rfecv.fit(xtrain_imputed, np.ravel(ytrain))

#storing metrics and model
pickle.dump(rfecv,open('logreg.pkl','w'))
