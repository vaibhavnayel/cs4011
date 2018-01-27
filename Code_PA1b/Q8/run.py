import numpy as np 
import matplotlib.pyplot as plt
from sklearn import metrics

#reading the data
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

def feedforward_batch_soft(P,p0,Q,q0,X):
    #forward pass of network with softmax output
    Z1=X.dot(P.T) + p0.T
    A1=sigmoid(Z1)
    Z2=A1.dot(Q.T)+ q0.T
    A2=softmax(Z2)
    return Z1,A1,Z2,A2

def feedforward_batch_sig(P,p0,Q,q0,X):
	#forward pass of network with sigmoid output
    Z1=X.dot(P.T) + p0.T
    A1=sigmoid(Z1)
    Z2=A1.dot(Q.T)+ q0.T
    A2=sigmoid(Z2)
    return Z1,A1,Z2,A2

def backprop_batch(X,Z1,A1,Z2,Y,P,p0,Q,q0,labels,loops):
    #implements backpropagation to find gradients of
    #weights and biases
    dLdy=labels-Y
    D1=(dLdy*((Y)*(1-Y)))
    dLdQ=D1.T.dot(A1)
    dLdq0=D1.T.dot(np.ones((len(A1),1)))
    D2=(D1.dot(Q))*(A1*(1-A1))
    dLdP=D2.T.dot(X)
    dLdp0=D2.T.dot(np.ones((len(X),1)))
    return np.array([dLdQ,dLdq0,dLdP,dLdp0])

def loss_sqr(Y,labels):
	#squared error loss
    return sum(sum((Y-labels)**2))/len(Y)

def loss_xent(Y,labels):
	#cross entropy loss
    return sum(sum(-labels*np.log(Y)))/len(Y)

def train(n_hid,stepsize,gamma,Loss):
    '''
    function to train neural network with 1 hidden layer with bias neurons
    n_hid= number of neurons in hidden layer
    stepsize= step size for training
    gamma= regularization parameter
    Loss= string 'CE' or 'SE' for cross-entropy and squared-error
    '''
    print 'hidden neurons: %d step size: %f gamma: %f'%(n_hid,stepsize,gamma)
    print 'Loss function: ', Loss
    np.random.seed(1)

    #initialize weights using uniform distribution from -1 to 1
    P=np.random.uniform(-1,1,size=(n_hid,96))
    p0=np.random.uniform(-1,1,size=(n_hid,1))
    Q=np.random.uniform(-1,1,size=(4,n_hid))
    q0=np.random.uniform(-1,1,size=(4,1))

    #run one forward pass to find initial error before training
    if Loss=='CE':
    	#using cross entropy
    	Z1,A1,Z2,Y= feedforward_batch_soft(P,p0,Q,q0,Xtrain)
    	print 'initial loss= ', loss_xent(Y,ytrain)  
    elif Loss=='SE':
    	#using square error
    	Z1,A1,Z2,Y= feedforward_batch_sig(P,p0,Q,q0,Xtrain)
    	print 'initial loss= ', loss_sqr(Y,ytrain)   

    #saving all errors for plotting
    test_errors=[]
    train_errors=[]
    test_accuracies=[]

    for i in range(100000):
        #backpropagate to find gradients
        dLdQ,dLdq0,dLdP,dLdp0 =backprop_batch(Xtrain,Z1,A1,Z2,Y,P,p0,Q,q0,ytrain,1)/len(Xtrain)
        
        #weight updates
        Q+=stepsize*dLdQ - gamma*Q/(len(Xtrain)**2)
        P+=stepsize*dLdP - gamma*P/(len(Xtrain)**2)
        p0+=stepsize*dLdp0 - gamma*p0/(len(Xtrain)**2)
        q0+=stepsize*dLdq0 - gamma*q0/(len(Xtrain)**2)

        #perform forward pass for next weight update
        if Loss=='CE':
        	Z1,A1,Z2,Y= feedforward_batch_soft(P,p0,Q,q0,Xtrain)
        elif Loss=='SE':
        	Z1,A1,Z2,Y= feedforward_batch_sig(P,p0,Q,q0,Xtrain)
        

        if i%1000==0 and Loss=='CE':
            #saving performance measures during training at regular intervals
            
            train_errors.append(loss_xent(Y,ytrain))
            z1,a1,z2,yhat= feedforward_batch_soft(P,p0,Q,q0,Xtest)
            test_errors.append(loss_xent(yhat,ytest))
            test_accuracies.append(sum(np.argmax(yhat,axis=1)==np.argmax(ytest,axis=1))/float(len(ytest)))
            print str(i)+' test error: %f train error: %f'%(test_errors[-1],train_errors[-1])
        elif i%1000==0 and Loss=='SE':
            #saving performance measures during training at regular intervals
            
            train_errors.append(loss_sqr(Y,ytrain))
            z1,a1,z2,yhat= feedforward_batch_sig(P,p0,Q,q0,Xtest)
            test_errors.append(loss_sqr(yhat,ytest))
            test_accuracies.append(sum(np.argmax(yhat,axis=1)==np.argmax(ytest,axis=1))/float(len(ytest)))
            print str(i)+' test error: %f train error: %f'%(test_errors[-1],train_errors[-1])
        
        #stopping condition
        #stop if test error increases consecutively 3 times.
        if i>20000 and test_errors[-1]>=test_errors[-2] and test_errors[-2]>=test_errors[-3]: break
    print 'final accuracy: ',sum(np.argmax(yhat,axis=1)==np.argmax(ytest,axis=1))/float(len(ytest))
    return [test_errors,train_errors,test_accuracies,yhat,P,Q,p0,q0]
'''
print 'Without regularization'
test_errors,train_errors,test_accuracies,yhat,P,Q,p0,q0=train(20,0.01,0,'CE')
m=metrics.precision_recall_fscore_support(np.argmax(ytest,axis=1),np.argmax(yhat,axis=1))
print 'precision: ',m[0]
print 'recall: ',m[1]
print 'F measure: ',m[2]
'''
print 'With regularization'
for g in [0,0.01,0.1,1,10,100]:

	test_errors,train_errors,test_accuracies,yhat,P,Q,p0,q0=train(40,0.01,g,'SE')
	m=metrics.precision_recall_fscore_support(np.argmax(ytest,axis=1),np.argmax(yhat,axis=1))
	print 'precision: ',m[0]
	print 'recall: ',m[1]
	print 'F measure: ',m[2]



