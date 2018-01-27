import numpy as np

data = np.genfromtxt('DS1-train.csv', delimiter=',')
xtrain=data[:,:-1]
ytrain=data[:,-1]
data = np.genfromtxt('DS1-test.csv', delimiter=',')
xtest=data[:,:-1]
ytest=data[:,-1]



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

print 'testing linear regression performance'
#learning weights
XTX=np.dot(xtrain.T,xtrain)
XTY=np.dot(xtrain.T,ytrain)
W=np.dot(np.linalg.inv(XTX),XTY)

#prediction step
yhat_linear=((np.dot(xtest,W)>0)-0.5)*2

linear_performance=measure(yhat_linear,ytest)
print 'weights learned:\n',W