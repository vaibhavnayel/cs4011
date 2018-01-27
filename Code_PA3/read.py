import os
import glob
import numpy as np
import scipy.sparse as sparse

try: del(mat)
except NameError: print 'ok'

def vectorise(l):
	vector=np.zeros(24748)
	for num in l:
		vector[num]+=1
	return sparse.lil_matrix(vector)
def readkfold(trainsetpath,testsetpath):
	try:del(legit_train,spam_train,legit_test,spam_test)
	except:pass
	for filename in glob.glob(os.path.join(trainsetpath, '*spm*.txt')):
		line=map(int,open(filename,'r').read().split()[1:])
		vector=vectorise(line)
		try:
			s_train=sparse.vstack((s_train,vector))
		except NameError:
			s_train=vector	

	for filename in glob.glob(os.path.join(testsetpath, '*spm*.txt')):
		line=map(int,open(filename,'r').read().split()[1:])
		vector=vectorise(line)
		try:
			s_test=sparse.vstack((s_test,vector))
		except NameError:
			s_test=vector	

	for filename in glob.glob(os.path.join(trainsetpath, '*legit*.txt')):
		line=map(int,open(filename,'r').read().split()[1:])
		vector=vectorise(line)
		try:
			l_train=sparse.vstack((l_train,vector))
		except NameError:
			l_train=vector	

	for filename in glob.glob(os.path.join(testsetpath, '*legit*.txt')):
		line=map(int,open(filename,'r').read().split()[1:])
		vector=vectorise(line)
		try:
			l_test=sparse.vstack((l_test,vector))
		except NameError:
			l_test=vector	

	return [l_train,s_train,l_test,s_test]

'''
path = '../PA3_datasets/2_NaiveBayes/set[12]/*'
for filename in glob.glob(os.path.join(path, '*.txt')):
	print filename
	line=map(int,open(filename,'r').read().split()[1:])
	vector=vectorise(line)
	try:
		mat=np.vstack((mat,vector))
	except NameError:
		mat=vector
'''


#multinomial


#K fold cross val
for i in range(1,6):

	print '\n'
	print 'fold:',i
	#reading data
	testsetpath='../PA3_datasets/2_NaiveBayes/set'+str(i)+'/*'
	trainsetpath='../PA3_datasets/2_NaiveBayes/set'+'[12345]'.replace(str(i),'')+'/*'
	[legit_train,spam_train,legit_test,spam_test]=readkfold(trainsetpath,testsetpath)

	#compute priors
	prior_spam=float(spam_train.shape[0])/(spam_train.shape[0]+legit_train.shape[0])
	prior_legit=float(legit_train.shape[0])/(spam_train.shape[0]+legit_train.shape[0])

	#compute term frequencies
	spam_freqs=spam_train.sum(axis=0)
	legit_freqs=legit_train.sum(axis=0)

	#compute smoothed probabilities
	spam_prob=(spam_freqs+1)/(spam_freqs +1).sum()
	legit_prob=(legit_freqs+1)/(legit_freqs+1).sum()

	#apply multinomial NB


	cla=np.hstack((np.log(spam_prob).T,np.log(legit_prob).T))
	spam_scores=spam_test.toarray().dot(cla)+np.log(np.array([prior_spam,prior_legit]))
	classes=np.argmax(spam_scores,axis=1)
	tn=classes.shape[0]-classes.sum()
	fp=classes.sum()

	legit_scores=legit_test.toarray().dot(cla)+np.log(np.array([prior_spam,prior_legit]))
	classes=np.argmax(legit_scores,axis=1)
	tp=classes.sum()
	fn=classes.shape[0]-classes.sum()

	acc=(float(tn+tp)/(tn+tp+fn+fp))
	prec=float(tp)/(tp+fp)
	rec= float(tp)/(tp+fn)
	f1=2*prec*rec/(prec+rec)
	print 'accuracy:',acc
	print 'recall:',rec
	print 'precision:',prec
	print 'f1 score:',f1accuracy: 0.977168949772
recall: 0.967479674797
precision: 0.991666666667
f1 score: 0.979423868313

ytrue=np.hstack((np.zeros(legit_scores.shape[0]),np.ones(spam_scores.shape[0])))
ypred=np.array(softmax(np.vstack((legit_scores,spam_scores)))[:,1].T)[0]

from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(ytrue, ypred,pos_label=1)
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')
plt.show()

#Bernoulli


#K fold cross val
for i in range(1,6):

	print '\n'
	print 'fold:',i
	#reading data
	testsetpath='../PA3_datasets/2_NaiveBayes/set'+str(i)+'/*'
	trainsetpath='../PA3_datasets/2_NaiveBayes/set'+'[12345]'.replace(str(i),'')+'/*'
	[legit_train,spam_train,legit_test,spam_test]=readkfold(trainsetpath,testsetpath)

	#compute priors
	prior_spam=float(spam_train.shape[0])/(spam_train.shape[0]+legit_train.shape[0])
	prior_legit=float(legit_train.shape[0])/(spam_train.shape[0]+legit_train.shape[0])

	#compute number of docs containing each word
	spam_freqs=(spam_train>0).sum(axis=0)
	legit_freqs=(legit_train>0).sum(axis=0)

	#compute smoothed probabilities
	spam_prob=(spam_freqs+1)/(2.+spam_train.shape[0])
	legit_prob=(legit_freqs+1)/(2.+legit_train.shape[0])

	#apply bernoulli NB
	cla1=np.hstack((np.log(spam_prob).T,np.log(legit_prob).T))
	cla2=np.hstack((np.log(1-spam_prob).T,np.log(1-legit_prob).T))
	
	spam_scores=(spam_test>0).toarray().dot(cla1) + (spam_test<1).toarray().dot(cla2)+np.log(np.array([prior_spam,prior_legit]))
	classes=np.argmax(spam_scores,axis=1)
	tn=classes.shape[0]-classes.sum()
	fp=classes.sum()	

	legit_scores=(legit_test>0).toarray().dot(cla1) + (legit_test<1).toarray().dot(cla2)+np.log(np.array([prior_spam,prior_legit]))
	classes=np.argmax((legit_test>0).toarray().dot(cla1) + (legit_test<1).toarray().dot(cla2)+np.log(np.array([prior_spam,prior_legit])),axis=1)
	tp=classes.sum()
	fn=classes.shape[0]-classes.sum()

	acc=(float(tn+tp)/(tn+tp+fn+fp))
	prec=float(tp)/(tp+fp)
	rec= float(tp)/(tp+fn)
	f1=2*prec*rec/(prec+rec)
	print 'accuracy:',acc
	print 'recall:',rec
	print 'precision:',prec
	print 'f1 score:',f1
	break
ytrue=np.hstack((np.zeros(legit_scores.shape[0]),np.ones(spam_scores.shape[0])))
ypred=np.array(softmax(np.vstack((legit_scores,spam_scores)))[:,1].T)[0]

from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(ytrue, ypred,pos_label=1)
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')
plt.show()


#beta bernoulli

#K fold cross val

p=[]
r=[]
f=[]
ac=[]
for i in range(1,6):
	print 'fold:',i
	#reading data
	testsetpath='../PA3_datasets/2_NaiveBayes/set'+str(i)+'/*'
	trainsetpath='../PA3_datasets/2_NaiveBayes/set'+'[12345]'.replace(str(i),'')+'/*'
	[legit_train,spam_train,legit_test,spam_test]=readkfold(trainsetpath,testsetpath)

	#compute priors
	a=1#alpha
	b=1#beta
	prior_spam=(float(spam_train.shape[0])+a-1)/(spam_train.shape[0]+legit_train.shape[0]+a+b-2)
	prior_legit=(float(legit_train.shape[0])+a-1)/(spam_train.shape[0]+legit_train.shape[0]+a+b-2)

	#compute number of docs containing each word
	spam_freqs=(spam_train>0).sum(axis=0)
	legit_freqs=(legit_train>0).sum(axis=0)

	#compute smoothed probabilities
	spam_prob=(spam_freqs+1)/(2.+spam_train.shape[0])
	legit_prob=(legit_freqs+1)/(2.+legit_train.shape[0])

	#apply bernoulli NB
	cla1=np.hstack((np.log(spam_prob).T,np.log(legit_prob).T))
	cla2=np.hstack((np.log(1-spam_prob).T,np.log(1-legit_prob).T))
	
	spam_scores=(spam_test>0).toarray().dot(cla1) + (spam_test<1).toarray().dot(cla2)+np.log(np.array([prior_spam,prior_legit]))
	classes=np.argmax(spam_scores,axis=1)
	tn=classes.shape[0]-classes.sum()
	fp=classes.sum()	

	legit_scores=(legit_test>0).toarray().dot(cla1) + (legit_test<1).toarray().dot(cla2)+np.log(np.array([prior_spam,prior_legit]))
	classes=np.argmax((legit_test>0).toarray().dot(cla1) + (legit_test<1).toarray().dot(cla2)+np.log(np.array([prior_spam,prior_legit])),axis=1)
	tp=classes.sum()
	fn=classes.shape[0]-classes.sum()

	acc=(float(tn+tp)/(tn+tp+fn+fp))
	ac.append(acc)
	prec=float(tp)/(tp+fp)
	p.append(prec)
	rec= float(tp)/(tp+fn)
	r.append(rec)
	f1=2*prec*rec/(prec+rec)
	f.append(f1)
	'''
	print 'accuracy:',acc
	print 'recall:',rec
	print 'precision:',prec
	print 'f1 score:',f1
	'''
print '(AVERAGE) accuracy: %f recall: %f precision: %f f1: %f'%(np.mean(ac),np.mean(r),np.mean(p),np.mean(f))

def softmax(x):
    e_x = np.exp(x/np.min(x,axis=1))*10 -20
    return e_x / e_x.sum(axis=1)
def softmax(x):
    e_x = np.exp(x-np.max(x,axis=1))
    return e_x / e_x.sum(axis=1)


def betabernoulli(a,b):
	print a,b
	p=[]
	r=[]
	f=[]
	ac=[]
	for i in range(1,6):
		#print 'fold:',i
		#reading data
		testsetpath='../PA3_datasets/2_NaiveBayes/set'+str(i)+'/*'
		trainsetpath='../PA3_datasets/2_NaiveBayes/set'+'[12345]'.replace(str(i),'')+'/*'
		[legit_train,spam_train,legit_test,spam_test]=readkfold(trainsetpath,testsetpath)

		#compute priors
		prior_spam=(float(spam_train.shape[0])+a-1)/(spam_train.shape[0]+legit_train.shape[0]+a+b-2)
		prior_legit=(float(legit_train.shape[0])+a-1)/(spam_train.shape[0]+legit_train.shape[0]+a+b-2)

		#compute number of docs containing each word
		spam_freqs=(spam_train>0).sum(axis=0)
		legit_freqs=(legit_train>0).sum(axis=0)

		#compute smoothed probabilities
		spam_prob=(spam_freqs+1)/(2.+spam_train.shape[0])
		legit_prob=(legit_freqs+1)/(2.+legit_train.shape[0])

		#apply bernoulli NB
		cla1=np.hstack((np.log(spam_prob).T,np.log(legit_prob).T))
		cla2=np.hstack((np.log(1-spam_prob).T,np.log(1-legit_prob).T))
		
		spam_scores=(spam_test>0).toarray().dot(cla1) + (spam_test<1).toarray().dot(cla2)+np.log(np.array([prior_spam,prior_legit]))
		classes=np.argmax(spam_scores,axis=1)
		tn=classes.shape[0]-classes.sum()
		fp=classes.sum()	

		legit_scores=(legit_test>0).toarray().dot(cla1) + (legit_test<1).toarray().dot(cla2)+np.log(np.array([prior_spam,prior_legit]))
		classes=np.argmax((legit_test>0).toarray().dot(cla1) + (legit_test<1).toarray().dot(cla2)+np.log(np.array([prior_spam,prior_legit])),axis=1)
		tp=classes.sum()
		fn=classes.shape[0]-classes.sum()

		acc=(float(tn+tp)/(tn+tp+fn+fp))
		ac.append(acc)
		prec=float(tp)/(tp+fp)
		p.append(prec)
		rec= float(tp)/(tp+fn)
		r.append(rec)
		f1=2*prec*rec/(prec+rec)
		f.append(f1)
		'''
		print 'accuracy:',acc
		print 'recall:',rec
		print 'precision:',prec
		print 'f1 score:',f1
		'''
		break
	print '(AVERAGE) accuracy: %f recall: %f precision: %f f1: %f'%(np.mean(ac),np.mean(r),np.mean(p),np.mean(f))
	ytrue=np.hstack((np.zeros(legit_scores.shape[0]),np.ones(spam_scores.shape[0])))
	ypred=np.array(softmax(np.vstack((legit_scores,spam_scores)))[:,1].T)[0]
	return (ac,r,p,f,ypred,ytrue)

(a,b,c,d,ypred,ytrue)=betabernoulli(100,500)

from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(ytrue, ypred,pos_label=1)
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')
plt.show()

for a in [0.00001,0.001,0.1,1,10,100,1000]:
	for b in [0.00001,0.001,0.1,1,10,100,1000]:
		betabernoulli(a,b)


def multidiric(a):
	print a
	p=[]
	r=[]
	f=[]
	ac=[]
	for i in range(1,6):
		#print 'fold:',i
		#reading data
		testsetpath='../PA3_datasets/2_NaiveBayes/set'+str(i)+'/*'
		trainsetpath='../PA3_datasets/2_NaiveBayes/set'+'[12345]'.replace(str(i),'')+'/*'
		[legit_train,spam_train,legit_test,spam_test]=readkfold(trainsetpath,testsetpath)

		#compute priors
		prior_spam=(float(spam_train.shape[0]))/(spam_train.shape[0]+legit_train.shape[0])
		prior_legit=(float(legit_train.shape[0]))/(spam_train.shape[0]+legit_train.shape[0])

		#compute number of docs containing each word
		spam_freqs=(spam_train>0).sum(axis=0)
		legit_freqs=(legit_train>0).sum(axis=0)

		#compute smoothed probabilities
		spam_prob=(spam_freqs+a-1.)/(np.sum(a)-len(a)+spam_train.shape[0])
		legit_prob=(legit_freqs+a-1.)/(np.sum(a)-len(a)+legit_train.shape[0])

		#apply bernoulli NB
		cla1=np.hstack((np.log(spam_prob).T,np.log(legit_prob).T))
		cla2=np.hstack((np.log(1-spam_prob).T,np.log(1-legit_prob).T))
		
		spam_scores=(spam_test>0).toarray().dot(cla1) + (spam_test<1).toarray().dot(cla2)+np.log(np.array([prior_spam,prior_legit]))
		classes=np.argmax(spam_scores,axis=1)
		tn=classes.shape[0]-classes.sum()
		fp=classes.sum()	

		legit_scores=(legit_test>0).toarray().dot(cla1) + (legit_test<1).toarray().dot(cla2)+np.log(np.array([prior_spam,prior_legit]))
		classes=np.argmax((legit_test>0).toarray().dot(cla1) + (legit_test<1).toarray().dot(cla2)+np.log(np.array([prior_spam,prior_legit])),axis=1)
		tp=classes.sum()
		fn=classes.shape[0]-classes.sum()

		acc=(float(tn+tp)/(tn+tp+fn+fp))
		ac.append(acc)
		prec=float(tp)/(tp+fp)
		p.append(prec)
		rec= float(tp)/(tp+fn)
		r.append(rec)
		f1=2*prec*rec/(prec+rec)
		f.append(f1)
		'''
		print 'accuracy:',acc
		print 'recall:',rec
		print 'precision:',prec
		print 'f1 score:',f1
		'''
		break
	print '(AVERAGE) accuracy: %f recall: %f precision: %f f1: %f'%(np.mean(ac),np.mean(r),np.mean(p),np.mean(f))
	ytrue=np.hstack((np.zeros(legit_scores.shape[0]),np.ones(spam_scores.shape[0])))
	ypred=np.array(softmax(np.vstack((legit_scores,spam_scores)))[:,1].T)[0]
	return (ac,r,p,f,ypred,ytrue)

(a,b,c,d,ypred,ytrue)=multidiric(200*np.ones((1,24748)))

from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(ytrue, ypred,pos_label=1)
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,color='b')
plt.show()




