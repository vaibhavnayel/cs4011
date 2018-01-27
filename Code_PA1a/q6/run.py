import numpy as np
import matplotlib.pyplot as plt


#Reading all 5 sets into a single array, indexed by first argument
Xtrain=[]
Ytrain=[]
Xtest=[]
Ytest=[]

for i in range(5):
	data = np.genfromtxt('CandC-train'+str(i+1)+'.csv', delimiter=',')
	Xtrain.append(data[:,:-1])
	Ytrain.append(data[:,-1])
	data = np.genfromtxt('CandC-test'+str(i+1)+'.csv', delimiter=',')
	Xtest.append(data[:,:-1])
	Ytest.append(data[:,-1])


#Finding weights for different regularization params l
AvgRSS_arr=[]#Saving all RSS for graph
for l in np.arange(0.5,2.5,0.05):
	AvgRSS=0
	#Averaging RSS over 5 sets
	for i in range(5):
		xtrain=Xtrain[i]
		ytrain=Ytrain[i]
		xtest=Xtest[i]
		ytest=Ytest[i]

		#Finding closed form solution
		XTX=np.dot(xtrain.T,xtrain)
		XTY=np.dot(xtrain.T,ytrain)
		I=np.eye(XTX.shape[0])
		I[0,0]=0 #Bias term is not regularized
		W=np.dot(np.linalg.inv(XTX+l*I),XTY)
		print W
		#Finding RSS
		yhat=xtest.dot(W)
		RSS=np.sum((yhat-ytest)**2)
		print 'RSS for set '+str(i+1)+': ',RSS
		AvgRSS+=RSS
	AvgRSS= AvgRSS/5
	print 'for l= '+str(l)+',RSS averaged over 5 sets= ',AvgRSS
	AvgRSS_arr.append(AvgRSS)
bestl=np.arange(0.5,2.5,0.05)[np.argmin(AvgRSS_arr)]
print 'best fit lambda= ',bestl
print 'best fit average RSS= ', min(AvgRSS_arr)
plt.plot(np.arange(0.5,2.5,0.05),AvgRSS_arr)
plt.title('Average RSS vs lambda')
plt.xlabel('lambda')
plt.ylabel('average RSS')
plt.show()