import math
import numpy as np
# Note: please don't add any new package, you should solve this problem using only the packages above.
#-------------------------------------------------------------------------
'''
    Problem 1: Linear Regression (Maximum Likelihood)
    In this problem, you will implement the linear regression method based upon maximum likelihood (least square).
    w'x + b = y
    You could test the correctness of your code by typing `nosetests -v test1.py` in the terminal.
    Note: please don't use any existing package for linear regression problem, implement your own version.
'''

#--------------------------
def compute_Phi(x,p):
    '''
        Compute the design matrix Phi of x for polynoial curve fitting problem. 
        We will construct p polynoials a the p features of the data samples. 
        The features of each sample, is x^0, x^1, x^2 ... x^(p-1)
        Input:
            x : a vector of samples in one dimensional space, a numpy vector of shape n by 1.
                Here n is the number of samples.
            p : the number of polynomials/features
        Output:
            Phi: the design/feature matrix of x, a numpy matrix of shape (n by p).
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    n = len(x)
    Phi = np.mat(np.ones((n,p)))
    column = np.mat(x)
    for i in range(1,p):
    	Phi[:,i] = np.multiply(Phi[:,i-1],column)
    #########################################
    return Phi 


#--------------------------
def least_square(Phi, y):
    '''
        Fit a linear model on training samples. Compute the paramter w using Maximum likelihood (equal to least square).
        Input:
            Phi: the design/feature matrix of the training samples, a numpy matrix of shape n by p
                Here n is the number of training samples, p is the number of features
            y : the sample labels, a numpy vector of shape n by 1.
        Output:
            w: the weights of the linear regression model, a numpy float vector of shape p by 1. 
        Hint: you could use np.linalg.inv() to compute the inverse of a matrix
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    w = np.linalg.inv(Phi.T * Phi) * Phi.T * y
    #########################################
    return w 



