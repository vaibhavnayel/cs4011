from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from mlxtend.feature_extraction import PrincipalComponentAnalysis as PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression,Perceptron,RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB


X = np.genfromtxt('../../contest_data/xtrain_linear_imputed.csv', delimiter=',')
print 'loaded X'
y = np.genfromtxt('../../contest_data/train.csv', delimiter=',')[1:,-1]
print 'loaded y'


pca = PCA(n_components=300)
X_pca = pca.fit(X).transform(X)
ada = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    learning_rate=1,
    random_state=0)
scores = cross_val_score(ada, X_pca, y,scoring='f1_micro',cv=5,verbose=5,n_jobs=1)

#22%


ada = AdaBoostClassifier(
    Perceptron(),
    n_estimators=100,
    learning_rate=1,
    random_state=0,algorithm='SAMME')
scores = cross_val_score(ada, X, y,scoring='f1_micro',cv=5,verbose=5,n_jobs=1)



ada = AdaBoostClassifier(
    RidgeClassifier(fit_intercept=True,solver='cholesky'),
    n_estimators=100,
    learning_rate=1,
    random_state=0,algorithm='SAMME')
scores = cross_val_score(ada, X, y,scoring='f1_micro',cv=5,verbose=5,n_jobs=1)

ada = AdaBoostClassifier(
    MultinomialNB(),
    n_estimators=100,
    learning_rate=1,
    random_state=0,algorithm='SAMME')
scores = cross_val_score(ada, X, y,scoring='f1_micro',cv=5,verbose=5,n_jobs=1)

