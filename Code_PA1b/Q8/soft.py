import numpy as np 
import matplotlib.pyplot as plt
from sklearn import metrics
Xtrain=np.genfromtxt('Train_features.csv',delimiter=',')
ytrain=np.genfromtxt('Train_labels.csv',delimiter=',')
Xtest=np.genfromtxt('Test_features.csv',delimiter=',')
ytest=np.genfromtxt('Test_labels.csv',delimiter=',')

#scaling columns to range [0,1]
Xtrain=Xtrain/Xtrain.max(axis=0)
Xtest=Xtest/Xtest.max(axis=0)

def sigmoid(x):
    return 1.0/(1+np.exp(-x))
def softmax(X):
    return np.exp(X)/np.sum(np.exp(X),axis=1).reshape((-1,1))

def feedforward_batch(P,p0,Q,q0,X):
    #forward pass of network
    Z1=X.dot(P.T) + p0.T
    A1=sigmoid(Z1)
    Z2=A1.dot(Q.T)+ q0.T
    A2=softmax(Z2)
    return Z1,A1,Z2,A2

def backprop_batch(X,Z1,A1,Z2,Y,P,p0,Q,q0,labels,loops):
    dLdy=labels-Y
    D1=(dLdy*((Y)*(1-Y)))
    dLdQ=D1.T.dot(A1)
    dLdq0=D1.T.dot(np.ones((len(A1),1)))
    D2=(D1.dot(Q))*(A1*(1-A1))
    dLdP=D2.T.dot(X)
    dLdp0=D2.T.dot(np.ones((len(X),1)))
    '''
    print dLdQ
    print dLdq0
    print dLdP
    print dLdp0
    '''
    return np.array([dLdQ,dLdq0,dLdP,dLdp0])

def loss(Y,labels):
    return sum(sum(-labels*np.log(Y)))/len(Y)

'''
n_hid=50
#np.random.seed(0)
P=np.random.uniform(-1,1,size=(n_hid,96))
p0=np.random.uniform(-1,1,size=(n_hid,1))
Q=np.random.uniform(-1,1,size=(4,n_hid))
q0=np.random.uniform(-1,1,size=(4,1))
Z1,A1,Z2,Y= feedforward_batch(P,p0,Q,q0,Xtrain)
print 'original loss= ', loss(Y,ytrain)
stepsize=0.01
l=0.0001
test_errors=[]
train_errors=[]
test_accuracies=[]
for i in range(100000):
    dLdQ,dLdq0,dLdP,dLdp0 =backprop_batch(Xtrain,Z1,A1,Z2,Y,P,p0,Q,q0,ytrain,1)/len(Xtrain)
    Q+=stepsize*dLdQ - l*Q
    P+=stepsize*dLdP - l*P
    p0+=stepsize*dLdp0 - l*p0
    q0+=stepsize*dLdq0 - l*q0
    Z1,A1,Z2,Y= feedforward_batch(P,p0,Q,q0,Xtrain)
    if i%1000==0:
        train_errors.append(loss(Y,ytrain))
        z1,a1,z2,yhat= feedforward_batch(P,p0,Q,q0,Xtest)
        test_errors.append(loss(yhat,ytest))
        print str(i)+' test error: %f train error: %f'%(test_errors[-1],train_errors[-1])
    if i>5000 and test_errors[-1]>test_errors[-2] and test_errors[-2]>test_errors[-3]: break
print 'accuracy: ',sum(np.argmax(yhat,axis=1)==np.argmax(ytest,axis=1))/float(len(ytest))
plt.plot(test_errors)
plt.plot(train_errors)
plt.show()
'''


def train(n_hid,stepsize,gamma):
    print 'hidden neurons: %d step size: %f gamma: %f'%(n_hid,stepsize,gamma)
    np.random.seed(1)
    P=np.random.uniform(-1,1,size=(n_hid,96))
    p0=np.random.uniform(-1,1,size=(n_hid,1))
    Q=np.random.uniform(-1,1,size=(4,n_hid))
    q0=np.random.uniform(-1,1,size=(4,1))
    Z1,A1,Z2,Y= feedforward_batch(P,p0,Q,q0,Xtrain)
    print 'original loss= ', loss(Y,ytrain)
    test_errors=[]
    train_errors=[]
    test_accuracies=[]
    for i in range(100000):
        dLdQ,dLdq0,dLdP,dLdp0 =backprop_batch(Xtrain,Z1,A1,Z2,Y,P,p0,Q,q0,ytrain,1)/len(Xtrain)
        Q+=stepsize*dLdQ - gamma*Q
        P+=stepsize*dLdP - gamma*P
        p0+=stepsize*dLdp0 - gamma*p0
        q0+=stepsize*dLdq0 - gamma*q0
        Z1,A1,Z2,Y= feedforward_batch(P,p0,Q,q0,Xtrain)
        if i%1000==0:
            train_errors.append(loss(Y,ytrain))
            z1,a1,z2,yhat= feedforward_batch(P,p0,Q,q0,Xtest)
            test_errors.append(loss(yhat,ytest))
            test_accuracies.append(sum(np.argmax(yhat,axis=1)==np.argmax(ytest,axis=1))/float(len(ytest)))
            print str(i)+' test error: %f train error: %f'%(test_errors[-1],train_errors[-1])
        if i>20000 and test_errors[-1]>test_errors[-2] and test_errors[-2]>test_errors[-3]: break
    print 'final accuracy: ',sum(np.argmax(yhat,axis=1)==np.argmax(ytest,axis=1))/float(len(ytest))
    return [test_errors,train_errors,test_accuracies,yhat,P,Q,p0,q0]
'''
test_errors,train_errors,test_accuracies,yhat,P,Q,p0,q0=train(30,0.01,0)
plt.plot(test_errors)
plt.plot(train_errors)
#plt.plot(test_accuracies)
plt.show()
'''

'''
min_errors=[]
max_accuracies=[]
for i in [5,10,15,20,30,40,50,60]:
    print 'for number of hidden neurons= '+str(i)
    test_errors,train_errors,test_accuracies,yhat,P,Q,p0,q0=train(i,0.01,0)
    min_errors.append(min(test_errors))
    max_accuracies.append(max(test_accuracies))
plt.plot([5,10,15,20,30,40,50,60],min_errors)
plt.plot([5,10,15,20,30,40,50,60],max_accuracies)
plt.legend(('minimum test error','maximum accuracy'))
plt.xlabel('number of hidden neurons')
plt.title('softmax output with cross entropy error')
plt.show()
'''

'''
#choosing best step size
min_errors=[]
max_accuracies=[]
for i in [0.0001,0.001,0.01,0.1,1.0,10]:
    print 'step size= '+str(i)
    test_errors,train_errors,test_accuracies,yhat,P,Q,p0,q0=train(20,i,0)
    min_errors.append(min(test_errors))
    max_accuracies.append(max(test_accuracies))
plt.plot([0.0001,0.001,0.01,0.1,1.0,10],min_errors)
plt.plot([0.0001,0.001,0.01,0.1,1.0,10],max_accuracies)
plt.legend(('minimum test error','maximum accuracy'))
plt.xlabel('step size')
plt.title('step size vs accuracy and loss')
plt.show()
'''

test_errors,train_errors,test_accuracies,yhat,P,Q,p0,q0=train(20,0.01,0)
metrics.precision_recall_fscore_support(np.argmax(ytest,axis=1),np.argmax(yhat,axis=1))
















