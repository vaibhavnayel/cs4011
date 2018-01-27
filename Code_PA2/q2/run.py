import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

#loading data
xtrain = np.genfromtxt('../../Data/DS3/train.csv', delimiter=',')
ytrain=(np.genfromtxt('../../Data/DS3/train_labels.csv', delimiter=',') -1.5)*2 #converting classes to +1 and -1 for convenience
xtest= np.genfromtxt('../../Data/DS3/test.csv', delimiter=',')
ytest=(np.genfromtxt('../../Data/DS3/test_labels.csv', delimiter=',') -1.5)*2

#findign LDA vector
lda = LinearDiscriminantAnalysis(solver='svd')
lda.fit(xtrain,ytrain)
b=lda.coef_
print "LDA vector obtained: ",b 

#projecting data onto LDA vector
xtrain_lda=xtrain.dot(b.T)
xtrain_lda=np.hstack((np.ones((len(xtrain_lda),1)),xtrain_lda))
xtest_lda=xtest.dot(b.T)
xtest_lda=np.hstack((np.ones((len(xtest_lda),1)),xtest_lda))

#saving projected data

np.savetxt("xtest_lda.csv", xtest_lda, delimiter=",")
np.savetxt("xtrain_lda.csv", xtrain_lda, delimiter=",")

#training linear classifier 
XTX=np.dot(xtrain_lda.T,xtrain_lda)
XTY=np.dot(xtrain_lda.T,ytrain)
W=np.dot(np.linalg.inv(XTX),XTY)
#evaluation
yhat=((np.dot(xtest_lda,W)>0)-0.5)*2

#metrics
print 'Metrics for linear classification with LDA:'
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
ax1.scatter(xtrain_lda[:1000,1],np.zeros(1000),c='orange',alpha=0.5)
ax1.scatter(xtrain_lda[1000:,1],np.zeros(1000),c='blue',alpha=0.5)
ax1.plot([-W[0]/W[1],-W[0]/W[1]],[-1,1],c='black')
ax1.set_xlabel('LDA vector')
ax1.set_title('projected xtrain plotted on x axis')
ax1.legend(['Decision boundary','class 1','class 2'])
ax2 = fig.add_subplot(212)
ax2.scatter(xtrain_lda[:1000,1],range(1000),c='orange',alpha=0.5)
ax2.scatter(xtrain_lda[1000:,1],range(1000),c='blue',alpha=0.5)
ax2.plot([-W[0]/W[1],-W[0]/W[1]],[0,1000],c='black')
ax2.set_xlabel('LDA vector')
ax2.set_ylabel('Index')
ax2.set_title('projected xtrain plotted by index for clarity')
ax2.legend(['Decision boundary','class 1','class 2'])
plt.show()
