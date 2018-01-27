import numpy as np
import matplotlib.pyplot as plt

#loading dataset
data = np.genfromtxt('DS1-train.csv', delimiter=',')
xtrain=data[:,:-1]
ytrain=np.asmatrix(data[:,-1]).T
data = np.genfromtxt('DS1-test.csv', delimiter=',')
xtest=data[:,:-1]
ytest=np.asmatrix(data[:,-1]).T

def knn(xtest,xtrain,ytrain,k):
	'''
	Returns classification of points in xtest 
	using votes from its k nearest neighbours 
	in xtrain
	Note: k must be odd
	'''
	Yhat=[]
	for row in xtest:
		
		#calculate list of distances from all training points
		a=xtrain-row
		dists=np.linalg.norm(a,axis=1)
		
		#find indices of closest k points
		indices=np.argsort(dists)[:k]
		
		#sign of sum of classes (-1 or 1) gives the class of data point
		if(sum(ytrain[indices])>0):
			classification=1
		else: classification=-1
		
		Yhat.append([classification])
	return np.asarray(Yhat,dtype='float64')

def measure(yhat,ytest):
	#function for returning performance measures for a model
	P=(ytest + yhat/2)
	tp=float(sum(P==1.5))#true positives
	tn=float(sum(P==-1.5))#true negatives
	fn=float(sum(P==0.5))#false negatives
	fp=float(sum(P==-0.5))#false positives

	accuracy=(tp+tn)/(tp+tn+fn+fp)
	precision=tp/(tp+fp)
	recall=tp/(tp+fn)
	F=2/((1/precision)+(1/recall))
	print 'accuracy= ', accuracy
	print 'precision= ', precision
	print 'recall= ', recall
	print 'F= ', F

	return [accuracy,precision,recall,F]

print 'testing knn'

knn_performance=[]

for i in range(1,51,2):
	yhat_knn=knn(xtest,xtrain,ytrain,i)
	print '\n'
	print 'k= ',i
	knn_performance.append(measure(yhat_knn,ytest))
knn_performance=np.asmatrix(knn_performance)
best=np.argmax(knn_performance[:,-1])
print '\n'
print 'best fit performance:'
print 'best fit k= ',best*2 +1
print 'best fit accuracy= ',knn_performance[best,0]
print 'best fit precision= ',knn_performance[best,1]
print 'best fit recall= ',knn_performance[best,2]
print 'best fit F measure= ',knn_performance[best,3]

plt.plot(range(1,51,2),knn_performance[:,-1])
plt.xlabel('k')
plt.ylabel('F-measure')
plt.title('F-measure vs k')
plt.show()






