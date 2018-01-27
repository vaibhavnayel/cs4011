import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

#running linear regression without dimensionality reduction
xtrain = np.genfromtxt('../../Data/DS3/train.csv', delimiter=',')
xtrain=np.hstack((np.ones((len(xtrain),1)),xtrain))#appending column of ones for bias
ytrain=(np.genfromtxt('../../Data/DS3/train_labels.csv', delimiter=',') -1.5)*2 #converting classes to +1 and -1 for convenience
xtest= np.genfromtxt('../../Data/DS3/test.csv', delimiter=',')
xtest=np.hstack((np.ones((len(xtest),1)),xtest))
ytest=(np.genfromtxt('../../Data/DS3/test_labels.csv', delimiter=',') -1.5)*2

#learning weights
XTX=np.dot(xtrain.T,xtrain)
XTY=np.dot(xtrain.T,ytrain)
W=np.dot(np.linalg.inv(XTX),XTY)

#evaluation
yhat=((np.dot(xtest,W)>0)-0.5)*2

#metrics
print 'Metrics for linear classification without dimensionality reduction:'
acc=sum(yhat==ytest)/float(len(ytest))
m=metrics.precision_recall_fscore_support(ytest,yhat,average='binary')
print 'Accuracy: ',acc
print 'Precision: ',m[0]
print 'Recall: ',m[1]
print 'F measure: ',m[2]
print 'Weights learned',W
print 'Boundary: %f + %fX + %fY + %fZ = 0'%(W[0],W[1],W[2],W[3])
print '\n'

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = np.arange(-2, 4, 1)
Y = np.arange(-5, 5, 1)
X, Y = np.meshgrid(X, Y)
Z=(-W[0]-X*W[1]-Y*W[2])/W[3]
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.scatter(xtrain[:1000,1],xtrain[:1000,2],xtrain[:1000,3],alpha=0.5)
ax.scatter(xtrain[1000:,1],xtrain[1000:,2],xtrain[1000:,3],alpha=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('DS3 3D plot')
#ax.legend(['class 1','class 2','Decision boundary'])

plt.show()


#performing PCA
from sklearn.decomposition import PCA

#finding principal component
pca = PCA(n_components=1)
pca.fit(xtrain[:,1:])
print 'Principal component learned: ',pca.components_

#projecting onto principal component
xtrain_pca=xtrain[:,1:].dot(pca.components_.T)
xtrain_pca=np.hstack((np.ones((len(xtrain_pca),1)),xtrain_pca))#adding bias column
xtest_pca=xtest[:,1:].dot(pca.components_.T)
xtest_pca=np.hstack((np.ones((len(xtest_pca),1)),xtest_pca))

#saving projected data
'''
np.savetxt("xtest_pca.csv", xtest_pca, delimiter=",")
np.savetxt("xtrain_pca.csv", xtrain_pca, delimiter=",")
'''

#training linear classifier 
XTX=np.dot(xtrain_pca.T,xtrain_pca)
XTY=np.dot(xtrain_pca.T,ytrain)
W=np.dot(np.linalg.inv(XTX),XTY)
#evaluation
yhat=((np.dot(xtest_pca,W)>0)-0.5)*2

print 'Metrics for linear classification with PCA:'
acc=sum(yhat==ytest)/float(len(ytest))
m=metrics.precision_recall_fscore_support(ytest,yhat)
print 'Accuracy: ',acc
print 'Precision: ',m[0]
print 'Recall: ',m[1]
print 'F measure: ',m[2]
print 'Weights learned',W
print 'Boundary: X=', -W[0]/W[1]
print '\n'

#plotting
fig = plt.figure(figsize=(7,9))
ax1 = fig.add_subplot(211)
ax1.scatter(xtrain_pca[:1000,1],np.zeros(1000),c='orange',alpha=0.5)
ax1.scatter(xtrain_pca[1000:,1],np.zeros(1000),c='blue',alpha=0.5)
ax1.plot([-W[0]/W[1],-W[0]/W[1]],[-1,1],c='black')
ax1.set_xlabel('Principal component')
ax1.set_title('projected xtrain plotted on x axis')
ax1.legend(['Decision boundary','class 1','class 2'])
ax2 = fig.add_subplot(212)
ax2.scatter(xtrain_pca[:1000,1],range(1000),c='orange',alpha=0.5)
ax2.scatter(xtrain_pca[1000:,1],range(1000),c='blue',alpha=0.5)
ax2.plot([-W[0]/W[1],-W[0]/W[1]],[0,1000],c='black')
ax2.set_xlabel('Principal component')
ax2.set_ylabel('Index')
ax2.set_title('projected xtrain plotted by index for clarity')
ax2.legend(['Decision boundary','class 1','class 2'])
plt.show()


