import os
import numpy as np
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from math import sqrt

def plotData(x, y,theta):
	fig = pyplot.figure()
	pyplot.plot(x, y, 'ro', ms=10, mec = 'k')
	pyplot.xlabel("X")
	pyplot.ylabel("y")

	if theta[0]!=-9999:
		pyplot.plot(x, np.dot(np.stack([np.ones(y.size), x], axis=1), theta), '-')
		pyplot.legend(['Training data', 'Linear Regression']);

	pyplot.show()

def plotResults(test_no, y_test, y_predictions):
	fig = pyplot.figure()
	pyplot.plot(test_no, y_test, 'ro', ms=10, mec = 'k')
	pyplot.plot(test_no, y_predictions, 'bo', ms=10, mec = 'k')
	pyplot.show()

def plotIteration(itr, J_history):
	itr = np.arange(itr)
	fig = pyplot.figure()
	pyplot.xlabel("Iteration")
	pyplot.ylabel("J (Cost)")
	pyplot.plot(itr, J_history, 'ro', ms=10, mec = 'k')
	pyplot.show()


def computeCost(X,y, theta):

	m = y.size

	'''hypo = np.dot(X,theta)
	sqDiff = np.square(hypo - y)
	J = 1/(2*m)*np.sum(sqDiff)'''

	J = 1/(2*m)*np.sum(np.square(np.dot(X,theta) - y))
	#J = np.sum((X.dot(theta) - y) ** 2)/(2 * m)

	return J

def featureNorm(X):
	mean = []
	std = []

	for i in range(X.shape[1]):
		col = X[:,i]

		mean.append(np.mean(col))
		std.append(np.std(col))

		X[:,i] = (col - np.mean(col))  / np.std(col)

	return mean, std, X	

def gradientDescent(X, y, theta, alpha, num_iters):
	m = len(y)
	J_history = []

	for i in range(num_iters):

		temp = np.dot(X, theta) - y
		X_new = np.dot(X.T, temp)
		theta = theta - (alpha/m)*X_new

		J_history.append(computeCost(X, y, theta))
	


	return theta, J_history

def LinearRegressionModel():

	trainData = np.loadtxt("boston_train.txt", delimiter = ',')

	X,y = trainData[:, 0:3], trainData[:, 3]

	#X = normalize(X)
	mu, sigma, X = featureNorm(X)



	#If you want to check the dimensions, use .shape attribute 
	m = len(y)
	#plotData(X,y,[-9999,-9999]) #-9999 for signifying no theta
	ones = np.ones((m,1))
	X = np.hstack((ones, X))


	theta=np.zeros(X.shape[1])
	alpha  = 0.1
	num_iters = 2000

	theta, J_history = gradientDescent(X, y, theta, alpha, num_iters)
	#plotIteration(num_iters, J_history)
	print('Cost : ',computeCost(X,y,theta))
	#print('\nTheta found by gradient descent:',theta)

	#plotData(X[:, 1], y,theta)
	'''temp = [1, 0.00632,18,2.31,0,0.538,6.575,65.2,4.09,1,296,15.3,396.9,4.98] #first example
	for i in range(len(temp)-1):		
		temp[i+1] = (temp[i] - mu[i])/sigma[i]'''

	testData = np.loadtxt("boston_test.txt", delimiter = ',')
	X_test, y_test = testData[:, 0:3], testData[:, 3]

	mu, sigma, X_test = featureNorm(X_test)
	#X_test = normalize(X_test)
	X_test = (X_test - np.mean(X_test))#/np.std(X_test)

	m = len(y_test)
	ones = np.ones((m,1))
	X_test = np.hstack((ones, X_test))

	predictions = np.dot(X_test, theta)
	#print('RMSE : ',np.sqrt((predictions - y_test)**2).mean())


	print("RMSE :",sqrt(mean_squared_error(y_test, predictions)))
	#plotResults(np.arange(333),y,predictions)


LinearRegressionModel()