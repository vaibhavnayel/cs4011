import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from mlxtend.feature_extraction import PrincipalComponentAnalysis as PCA
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
#load datav
xtrain = np.genfromtxt('../../contest_data/train.csv', delimiter=',')[1:,1:-1]
xtest = np.genfromtxt('../../contest_data/test.csv', delimiter=',')[1:,1:]

'''
ytrain = np.genfromtxt('../../contest_data/train.csv', delimiter=',')[1:,-1]
pca = PCA(n_components=300)
X_pca = pca.fit(X).transform(xtrain[:,500:])
dt=DecisionTreeRegressor(max_depth=5)

svr=SVR(C=1, epsilon=0.001)
et = ExtraTreesRegressor(n_estimators=100, max_depth=None, random_state=0,verbose=5)
'''

#imputing training data
lin=LinearRegression()
for i  in range(500):
	print i
	missings=np.isnan(xtrain[:,i])
	nonmissings=~missings
	xtrain[missings,i]=lin.fit(X_pca[nonmissings,:],xtrain[nonmissings,i]).predict(X_pca[missings,:])
	#scores = cross_val_score(lin, X_pca[nonmissings,:], xtrain[nonmissings,i],scoring='neg_mean_absolute_error',cv=5,verbose=0)
	#print scores


#decision trees
xtrain = np.genfromtxt('../../contest_data/train.csv', delimiter=',')[1:,1:-1]
ytrain = np.genfromtxt('../../contest_data/train.csv', delimiter=',')[1:,-1]
dt=DecisionTreeRegressor()
for i  in range(500):
	print i
	missings=np.isnan(xtrain[:,i])
	nonmissings=~missings
	xtrain[missings,i]=dt.fit(xtrain[nonmissings,500:],xtrain[nonmissings,i]).predict(xtrain[missings,500:])
	#scores = cross_val_score(dt, X_pca[nonmissings,:], xtrain[nonmissings,i],scoring='neg_mean_absolute_error',cv=5,verbose=5)
	#print scores

np.savetxt('../../contest_data/xtest_tree_imputed.csv',xtrain,delimiter=',')

from sklearn.naive_bayes import MultinomialNB
gnb = MultinomialNB()
scores = cross_val_score(gnb, xtrain, ytrain,scoring='f1_micro',cv=5,verbose=5)
print scores.mean()



pca = PCA(n_components=300)
xtrain_pca = pca.fit(xtrain[:,500:]).transform(xtrain[:,500:])
xtest_pca = pca.fit(xtrain[:,500:]).transform(xtest[:,500:])
#imputing test data
for i  in range(500):
	print i
	train_missings=np.isnan(xtrain[:,i])
	train_nonmissings=~train_missings
	test_missings=np.isnan(xtest[:,i])
	test_nonmissings=~test_missings
	xtest[test_missings,i]=lin.fit(xtrain_pca[train_nonmissings,:],xtrain[train_nonmissings,i]).predict(xtest_pca[test_missings,:])

np.savetxt('../../contest_data/xtest_linear_imputed.csv',xtest,delimiter=',')

'''
Best imputer was found to be a linear model as all the others overfit.

TO DO: plot class conditioned densities before and after imputation
'''



'''
pca = PCA(n_components=300)
X_pca = pca.fit(X).transform(xtrain[:,500:])
parameters = { 'C':np.logspace(-3,3,7),'gamma':np.logspace(-4,2,7),'epsilon':np.logspace(-3,1,5)}
svr = SVR()

for i  in range(2):
	print 'column',i
	missings=np.isnan(xtrain[:,i])
	nonmissings=~missings	
	mod = GridSearchCV(svr, parameters,cv=5,scoring='neg_mean_absolute_error',verbose=5)
	mod.fit(X_pca[nonmissings,:],xtrain[nonmissings,i])
	print 'best accuracy: ',mod.best_score_
	print 'best parameters: ',mod.best_estimator_

'''