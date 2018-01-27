import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors.kde import KernelDensity

#load data
xtrain = np.genfromtxt('../../contest_data/train.csv', delimiter=',')[1:,1:-1]
ytrain = np.genfromtxt('../../contest_data/train.csv', delimiter=',')[1:,-1]
ytrain=np.asmatrix(ytrain).T
xtrain_linear_imputed=np.genfromtxt('../../contest_data/xtrain_linear_imputed.csv', delimiter=',')

#imputing by sampling from class conditioned density estimate
#class conditional density estimate of column 1
for k in range(500):
	finite=np.isfinite(xtrain[:,k])
	nans=np.isnan(xtrain[:,k])
	y=np.array(ytrain[finite].T)
	X=xtrain[finite,k][:,np.newaxis]
	print k
	for i in range(29):
		#X_plot=np.linspace(0,1,1000)[:,np.newaxis]
		ind=y==float(i)
		kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(X[ind[0]])
		nans_i=np.isnan(xtrain[:,k])*np.array((ytrain==float(i)).T)
		xtrain[nans_i[0],k]=np.array(kde.sample(sum(nans_i[0]),random_state=0).T)
		log_dens = kde.score_samples(X_plot)
		#dens=np.exp(log_dens)
		#plt.plot(X_plot,dens)
	#plt.show()
