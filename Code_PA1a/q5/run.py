import numpy as np


'''
#Creating 5 train-test sets
data= np.genfromtxt('CandC.csv', delimiter=',')
for i in range(5):
	A=np.random.permutation(data)
	test=A[:400,:]
	train=A[400:,:]
	np.savetxt('CandC-test'+str(i+1)+'.csv',test,delimiter=',')
	np.savetxt('CandC-train'+str(i+1)+'.csv',train,delimiter=',')
'''

SumRSS=0
SumAvgRSS=0
for i in range(5):
	#Reading Training and Test data
	data = np.genfromtxt('CandC-train'+str(i+1)+'.csv', delimiter=',')
	xtrain=data[:,:-1]
	ytrain=data[:,-1]

	data = np.genfromtxt('CandC-test'+str(i+1)+'.csv', delimiter=',')
	xtest=data[:,:-1]
	ytest=data[:,-1]

	#Finding Closed form solution
	XTX=np.dot(xtrain.T,xtrain)
	XTY=np.dot(xtrain.T,ytrain)
	W=np.dot(np.linalg.inv(XTX),XTY)
    #np.savetxt('Coeffs'+str(i+1)+'.csv',W,delimiter=',')
	print 'Weights learned for set '+ str(i+1)+'\n', W

	yhat=xtest.dot(W)
	RSS=np.sum((yhat-ytest)**2)
	print 'RSS for set '+str(i+1)+': ',RSS
	mean_RSS=np.mean((yhat-ytest)**2)
	print 'average RSS for set '+str(i+1)+': ',mean_RSS
	SumRSS+=RSS
AvgRSS= SumRSS/5
print 'RSS averaged over 5 sets: ',AvgRSS
