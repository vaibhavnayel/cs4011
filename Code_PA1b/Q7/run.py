import numpy as np
from sklearn import metrics
ytest = np.genfromtxt('Test_labels.csv', delimiter=',')
'''
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
'''
p=['0','0001','001','01']
#read models and report performance
for i in range(4):
	yhat=np.genfromtxt('result'+p[i]+'.csv', delimiter=',')
	print '\n\n'
	print 'for regularization parameter ='+str(p[i])
	m=metrics.precision_recall_fscore_support(ytest,yhat)
	print m[0]
	print m[1]
	print m[2]