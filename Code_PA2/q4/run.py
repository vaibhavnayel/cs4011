from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib import colors
from  sklearn.svm import SVC
from mpl_toolkits.mplot3d import Axes3D
import pickle

#loading data
xtrain = np.genfromtxt('../../Data/DS2/Train_features.csv', delimiter=',')
#xtrain=xtrain/np.asmatrix(np.sum(xtrain,axis=1)).T
xtrain=xtrain/np.asmatrix(np.max(xtrain,axis=1)).T #scaling
ytrain = np.genfromtxt('../../Data/DS2/Train_labels.csv', delimiter=',')
ytrain=np.argmax(ytrain,axis=1)

xtest = np.genfromtxt('../../Data/DS2/Test_features.csv', delimiter=',')
#xtest=xtest/np.asmatrix(np.sum(xtest,axis=1)).T
xtest=xtest/np.asmatrix(np.max(xtest,axis=1)).T #scaling
ytest = np.genfromtxt('../../Data/DS2/Test_labels.csv', delimiter=',')
ytest=np.argmax(ytest,axis=1)



#linear kernel
print'Testing linear kernel'

acc=[]
for c in np.logspace(-2,2,20):
	#initializing model
	lin=SVC(C=c,kernel='linear',random_state=1) 
	#cross validating using stratified 5-fold splits
	a=np.mean(cross_val_score(lin,xtrain,ytrain,scoring='accuracy',cv=5))
	print 'for C = %f average accuracy over 5 folds: %f'%(c,a) 
	acc.append(a)

#plotting
plt.plot(np.logspace(-2,2,20),acc)
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('accuracy')
plt.title('accuracy averaged over 5 splits vs C\n linear kernel')
#plt.savefig('linear.png')
plt.show()

best_C=np.logspace(-2,2,20)[np.argmax(acc)]
print 'highest accuracy: '+str(max(acc))
print 'best C: '+ str(best_C)

#finding test set accuracy
clf=SVC(C=best_C,kernel='linear',random_state=1)
clf.fit(xtrain,ytrain)
p=clf.predict(xtest)
a=sum(p==ytest)/float(len(ytest))
print 'Test set accuracy: ',a

pickle.dump(clf,open('svm_model1.pkl','w'))

#RBF kernel
print 'Testing RBF kernel'
#generating parameter matrix for plotting and exhaustive grid search
xx, yy = np.meshgrid(np.logspace(-1,1,10),np.logspace(-2,0,10))
p=(np.vstack((xx.flatten(), yy.flatten()))).T
acc=[]
best_acc=0
best_params=[0,0]
for i in p:
	#initialize model
	rbf = SVC(C=i[0],gamma=i[1],kernel='rbf',random_state=1,max_iter=100000)
	#5 fold stratified cross validation
	a=np.mean(cross_val_score(rbf,xtrain,ytrain,scoring='accuracy',cv=5,verbose=0))
	#store metric
	acc.append(a)
	print 'for C = %f and Gamma= %f, average accuracy over 5 folds: %f'%(i[0],i[1],a) 
	if a>best_acc:
		best_acc=a
		best_params=i

acc = np.asarray(acc).reshape(xx.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.bar(np.logspace(-1,1,10), acc, zs=np.logspace(-2,0,10), zdir='y', alpha=0.8)

X,Y=np.meshgrid(np.linspace(-1,1,10),np.linspace(-2,0,10))
surf = ax.plot_surface(X, Y, acc, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel('C (log)')
ax.set_ylabel('Gamma (log)')
ax.set_zlabel('Accuracy')
ax.set_title('Accuracy vs C and Gamma surface plot\n RBF kernel')
plt.show()

print 'highest accuracy: ',best_acc
print 'best C: %f, best Gamma: %f'%(best_params[0],best_params[1])

#finding test accuracy
clf=SVC(C=best_params[0],gamma=best_params[1],kernel='rbf',random_state=1)
clf.fit(xtrain,ytrain)
p=clf.predict(xtest)
a=sum(p==ytest)/float(len(ytest))
print 'Test set accuracy: ',a

#pickle.dump(clf,open('svm_model3.pkl','w'))

'''
parameters = { 'C':np.logspace(-1,1,10),'gamma':np.logspace(-2,0,10)}
svc = SVC(kernel='rbf',random_state=1)
rbf = GridSearchCV(svc, parameters,cv=5,scoring='accuracy',verbose=0)
rbf.fit(xtrain,ytrain)
print 'best accuracy: ',rbf.best_score_
print 'best parameters: ',rbf.best_estimator_

'''

#sigmoid kernel
print 'Testing sigmoid kernel'
#generating parameter matrix for plotting and exhaustive grid search
xx, yy = np.meshgrid(np.logspace(4,8,5),np.logspace(-8,-4,4))
p=(np.vstack((xx.flatten(), yy.flatten()))).T
acc=[]
best_acc=0
best_params=[0,0]
for i in p:
	#initialize model
	sig = SVC(C=i[0],gamma=i[1],kernel='sigmoid',random_state=1,max_iter=100000)
	#5 fold stratified cross validation
	a=np.mean(cross_val_score(sig,xtrain,ytrain,scoring='accuracy',cv=5,verbose=0))
	#store metric
	acc.append(a)
	print 'for C = %f and Gamma= %f, average accuracy over 5 folds: %f'%(i[0],i[1],a) 
	if a>best_acc:
		best_acc=a
		best_params=i

acc = np.asarray(acc).reshape(xx.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.bar(np.logspace(-1,1,10), acc, zs=np.logspace(-2,0,10), zdir='y', alpha=0.8)

X,Y=np.meshgrid(np.linspace(4,8,5),np.linspace(-8,-4,4))
surf = ax.plot_surface(X, Y, acc, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.set_xlabel('C (log)')
ax.set_ylabel('Gamma (log)')
ax.set_zlabel('Accuracy')
ax.set_title('Accuracy vs C and Gamma surface plot\n sigmoid kernel')
plt.show()

'''
extent=[1,7,-7,3]
plt.title('Accuracy heat map')
plt.ylabel('Gamma')
plt.xlabel('C')
plt.imshow(acc, extent=extent)
plt.show()
'''
print 'highest accuracy: ',best_acc
print 'best C: %f, best Gamma: %f'%(best_params[0],best_params[1])

#finding test accuracy
clf=SVC(C=best_params[0],gamma=best_params[1],kernel='sigmoid',random_state=1)
clf.fit(xtrain,ytrain)
p=clf.predict(xtest)
a=sum(p==ytest)/float(len(ytest))
print 'Test set accuracy: ',a

#pickle.dump(clf,open('svm_model4.pkl','w'))
'''
parameters = { 'C':np.logspace(1,3,12),'gamma':np.logspace(-4,-2,12)}
svc = SVC(kernel='sigmoid',random_state=1)
sig = GridSearchCV(svc, parameters,cv=5,scoring='accuracy',verbose=3)
sig.fit(xtrain,ytrain)
print 'best accuracy: ',sig.best_score_
print 'best parameters: ',sig.best_estimator_
'''
print 'Testing polynomial kernel'

#generating parameter matrix for plotting and exhaustive grid search
xx, yy = np.meshgrid(np.logspace(2,8,23),range(2,6))
p=(np.vstack((xx.flatten(), yy.flatten()))).T
acc=[]
best_acc=0
best_params=[0,0]
for i in p:
	#initialize model
	sig = SVC(C=i[0],degree=i[1],kernel='poly',random_state=1,max_iter=100000)
	#5 fold stratified cross validation
	a=np.mean(cross_val_score(sig,xtrain,ytrain,scoring='accuracy',cv=5,verbose=0))
	#store metric
	acc.append(a)
	print 'for C = %f and degree= %f, average accuracy over 5 folds: %f'%(i[0],i[1],a) 
	if a>best_acc:
		best_acc=a
		best_params=i

acc = np.asarray(acc).reshape(xx.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.bar(np.logspace(-1,1,10), acc, zs=np.logspace(-2,0,10), zdir='y', alpha=0.8)

X,Y=np.meshgrid(np.linspace(2,8,23),range(2,6))
surf = ax.plot_surface(X, Y, acc, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.set_xlabel('C (log)')
ax.set_ylabel('degree')
ax.set_zlabel('Accuracy')
ax.set_title('Accuracy vs C and degree surface plot\n polynomial kernel')
plt.show()

'''
extent=[1,7,-7,3]
plt.title('Accuracy heat map')
plt.ylabel('Gamma')
plt.xlabel('C')
plt.imshow(acc, extent=extent)
plt.show()
'''
print 'highest accuracy: ',best_acc
print 'best C: %f, best degree: %f'%(best_params[0],best_params[1])

#finding test accuracy
clf=SVC(C=best_params[0],degree=best_params[1],kernel='poly',random_state=1)
clf.fit(xtrain,ytrain)
p=clf.predict(xtest)
a=sum(p==ytest)/float(len(ytest))
print 'Test set accuracy: ',a
'''
parameters = { 'C':np.logspace(-4,-1,4),'gamma':np.logspace(0,3,4),'degree':[2,3,4]}
svc = SVC(kernel='poly',random_state=1)
poly = GridSearchCV(svc, parameters,cv=5,scoring='accuracy',verbose=3)
poly.fit(xtrain,ytrain)
print 'best accuracy: ',poly.best_score_
print 'best parameters: ',poly.best_estimator_
p=poly.predict(xtest)
a=sum(p==ytest)/float(len(ytest))
print 'Test set accuracy: ',a
'''
#pickle.dump(clf,open('svm_model2.pkl','w'))

