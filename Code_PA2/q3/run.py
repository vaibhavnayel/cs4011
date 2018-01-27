import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

X = np.genfromtxt('../../Data/iris/iris.csv', delimiter=',')[:,:-1]
Y = np.genfromtxt('../../Data/iris/iris.csv', delimiter=',')[:,-1]

plt.scatter(X[:50,0],X[:50,1])
plt.scatter(X[50:100,0],X[50:100,1])
plt.scatter(X[100:,0],X[100:,1])
plt.legend(['class 1','class 2','class 3'])
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.title('Iris dataset')
plt.show()

#performing LDA
lda = LinearDiscriminantAnalysis(solver='svd')
lda.fit(X,Y)

#plotting LDA decision boundary
plt.scatter(X[:,0],X[:,1])
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
plt.clf()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))
Z = lda.predict((np.vstack((xx.flatten(), yy.flatten()))).T)
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z,cmap='Set3')
plt.contour(xx,yy,Z, linewidths=1, colors='k')
plt.scatter(X[:50,0],X[:50,1])
plt.scatter(X[50:100,0],X[50:100,1])
plt.scatter(X[100:,0],X[100:,1])
plt.legend(['class 1','class 2','class 3'])
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.title('LDA decision boundaries')
plt.show()

#performing QDA
qda = QuadraticDiscriminantAnalysis()
qda.fit(X,Y)

#plotting QDA decision boundary
plt.scatter(X[:,0],X[:,1])
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
plt.clf()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))
Z = qda.predict((np.vstack((xx.flatten(), yy.flatten()))).T)
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z,cmap='Set3')
plt.contour(xx,yy,Z, linewidths=1, colors='k')
plt.scatter(X[:50,0],X[:50,1])
plt.scatter(X[50:100,0],X[50:100,1])
plt.scatter(X[100:,0],X[100:,1])
plt.legend(['class 1','class 2','class 3'])
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.title('QDA decision boundaries')
plt.show()



#performing RDA
for i in np.logspace(-4,2,16):
	rda = QuadraticDiscriminantAnalysis(reg_param=i)
	rda.fit(X,Y)

	#plotting QDA decision boundary
	plt.scatter(X[:,0],X[:,1])
	x_min, x_max = plt.xlim()
	y_min, y_max = plt.ylim()
	plt.clf()
	xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
	                     np.linspace(y_min, y_max, 500))
	Z = rda.predict((np.vstack((xx.flatten(), yy.flatten()))).T)
	Z = Z.reshape(xx.shape)
	plt.pcolormesh(xx, yy, Z,cmap='Set3')
	plt.contour(xx,yy,Z, linewidths=1, colors='k')
	plt.scatter(X[:50,0],X[:50,1])
	plt.scatter(X[50:100,0],X[50:100,1])
	plt.scatter(X[100:,0],X[100:,1])
	plt.legend(['class 1','class 2','class 3'])
	plt.xlabel('feature 1')
	plt.ylabel('feature 2')
	plt.title('RDA decision boundaries \n regularization parameter= '+str(i))
	plt.savefig(str(np.log10(i))+'gif.png')
	#plt.show()














