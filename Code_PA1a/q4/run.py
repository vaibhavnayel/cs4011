import numpy as np
np.random.seed(0)

#Reading and preprocessing data
f=open('CandC_raw.txt','r')
t=f.readlines()
t1=[r[:-2] for r in t]
t2=[r.split(',') for r in t1]
t3=np.asarray(t2,dtype='float64')
t4=np.hstack((np.ones([1994,1]),t3))

#Extracting indices of missing values
missingcols=np.asarray(np.where(t4[1,:]==-1))
missingrows=np.asarray(np.where(t4[:,122]==-1))
knowncols=np.asarray(np.where(t4[1,:]!=-1))
knownrows=np.asarray(np.where(t4[:,122]!=-1))

#Selecting data for filling missing values

#P is the known rows of columns without missing values
#q is the known rows of the columns with missing values
#We will treat q as the output vector and Q as data
P=t4[knownrows,:][0][:,knowncols][:,0,:-1]
q=t4[knownrows,:][0][:,missingcols][:,0,:]

#Taking train:test==2:1
Ptest=P[:100,:]
Ptrain=P[100:,:]
qtest=q[:100,:]
qtrain=q[100:,:]

#Finding closed form solution
PTP=np.dot(Ptrain.T,Ptrain)
PTQ=np.dot(Ptrain.T,qtrain)
W=np.dot(np.linalg.inv(PTP),PTQ)
qhat=np.dot(Ptest,W)
col_mse=np.mean((qhat-qtest)*(qhat-qtest),axis=0)
mse=np.mean(col_mse,axis=0)

#reconstructing the data 
X=t4[missingrows,:][0][:,knowncols][:,0,:-1]

missing_estimates=X.dot(W)
SD=(np.var(t4[knownrows,:][0],axis=0)[missingcols])**0.5
noise=np.random.randn(missing_estimates.shape[0],missing_estimates.shape[1])
missing_estimates+=noise*SD

X=np.hstack((X,missing_estimates))#1675 rows. purely reconstructed
X1=np.hstack((P,q))#319 rows. purely known 
X=np.vstack((X,X1))#completed 
#reordering y array to match X
y=np.vstack((t4[missingrows,-1].T,t4[knownrows,-1].T))

'''
Saving completed dataset
np.savetxt('CandC.csv',np.random.permutation(np.hstack((X,y))),delimiter=',')
'''











