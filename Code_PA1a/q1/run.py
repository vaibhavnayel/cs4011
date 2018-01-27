import numpy as np
np.random.seed(0)

#Data Synthesis
dim=20 #Number of columns
s=0.5 #Shift parameter for shifting centroid
mean1=np.zeros(dim)
mean2=mean1+s*np.random.randn(dim)

#Creating a symmetric positive semidefinite covariance matrix:
#It is known that A*A' has this property as long as A is full rank
#In general, the random function will create linearly independent columns
#Therefore this is a reasonable choice for creating the covariance matrix
A=(np.random.randn(dim,dim))
cov=A.dot(A.T)

x1 = np.random.multivariate_normal(mean1, cov, 2000)
x2 = np.random.multivariate_normal(mean2, cov, 2000)

x1=np.hstack((np.ones([2000,1]),x1))
x2=np.hstack((np.ones([2000,1]),x2))

#Taking labels as 1 and -1
y1=-np.ones((2000,1))
y2=np.ones((2000,1))

#Splitting data into test and train sets
xtest=np.vstack((x1[:600,:],x2[:600,:]))
ytest=np.vstack((y1[:600,:],y2[:600,:]))

xtrain=np.vstack((x1[600:,:],x2[600:,:]))
ytrain=np.vstack((y1[600:,:],y2[600:,:]))

#Dataset saving code commented out since 
#data is already saved
'''
np.savetxt("DS1-train.csv", np.hstack((xtrain,ytrain)), delimiter=",")
np.savetxt("DS1-test.csv", np.hstack((xtest,ytest)), delimiter=",")
'''

print 'centroid of class 1:\n',mean1
print 'centroid of class 2:\n',mean2
print 'covariance matrix:\n',cov














