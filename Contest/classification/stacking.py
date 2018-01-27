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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from mlxtend.classifier import StackingClassifier


X = np.genfromtxt('../../contest_data/xtrain_linear_imputed.csv', delimiter=',')
print 'loaded X'
y = np.genfromtxt('../../contest_data/train.csv', delimiter=',')[1:,-1]
print 'loaded y'

pca = PCA(n_components=300)
X_pca = pca.fit(X).transform(X)

et = ExtraTreesClassifier(n_estimators=1000, max_depth=None, random_state=0,verbose=0)
svc=SVC(C=1,gamma='auto',verbose=0)
#dt = DecisionTreeClassifier(min_samples_leaf=5,random_state=0)
rf = RandomForestClassifier(n_estimators=1000, max_depth=None, random_state=0,verbose=0)
lr = LogisticRegression()
sclf = StackingClassifier(classifiers=[et, svc,rf],
                          use_probas=True,
                          average_probas=False,
                          meta_classifier=lr)
scores = cross_val_score(bag, X, y,scoring='f1_micro',cv=5,verbose=5,n_jobs=-1)
print scores