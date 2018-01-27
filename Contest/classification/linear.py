import numpy as np
import matplotlib.pyplot as plt

xtrain = np.genfromtxt('../../contest_data/train.csv', delimiter=',')[1:,1:-1]
xtrain_imputed = np.genfromtxt('../../contest_data/xtrain_density_imputed.csv', delimiter=',')
ytrain = np.genfromtxt('../../contest_data/train.csv', delimiter=',')[1:,-1]
ytrain=np.asmatrix(ytrain).T


#SVM for recursive feature selection
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

# Create the RFE object and compute a cross-validated score.
svc = SVC(kernel="linear")
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=svc, step=400, cv=StratifiedKFold(2),
              scoring='accuracy',verbose=10)
rfecv.fit(xtrain_imputed, np.ravel(ytrain))

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation accuracy")
plt.title('accuracy vs dropped features (SVM)')
plt.plot([0]+range(200,2700,400), rfecv.grid_scores_)
#plt.show()
plt.savefig('svm_dropped_features.png')
pickle.dump(rfecv,open('logreg.pkl','w'))
np.savetxt('gridscores.csv',rfecv.grid_scores_,delimiter=',')
np.savetxt('gridscores.csv',rfecv.ranking_,delimiter=',')
np.savetxt('gridscores.csv',rfecv.support_,delimiter=',')
np.savetxt('gridscores.csv',rfecv.n_features_,delimiter=',')
