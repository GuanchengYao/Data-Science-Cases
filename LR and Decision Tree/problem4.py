import math
import numpy as np
from problem3 import Tree
#-------------------------------------------------------------------------
'''
    Problem 4: Decision Tree (with continous attributes)
    In this problem, you will implement the decision tree method for classification problems.
    You could test the correctness of your code by typing `nosetests -v test4.py` in the terminal.
'''

#--------------------------
class Node:
    '''
        Decision Tree Node (with continous attributes)
        Inputs: 
            X: the data instances in the node, a numpy matrix of shape p by n.
               Each element can be int/float.
               Here n is the number data instances in the node, p is the number of attributes.
            Y: the class labels, a numpy array of length n.
               Each element can be int/float/string.
            i: the index of the attribute being tested in the node, an integer scalar 
            th: the threshold on the attribute, a float scalar.
            C1: the child node for values smaller than threshold
            C2: the child node for values larger than threshold
            isleaf: whether or not this node is a leaf node, a boolean scalar
            p: the label to be predicted on the node (i.e., most common label in the node).
    '''
    def __init__(self,X=None,Y=None, i=None,th=None,C1=None, C2=None, isleaf= False,p=None):
        self.X = X
        self.Y = Y
        self.i = i
        self.th = th 
        self.C1= C1
        self.C2= C2
        self.isleaf = isleaf
        self.p = p


#-----------------------------------------------
class DT(Tree):
    '''
        Decision Tree (with contineous attributes)
        Hint: DT is a subclass of Tree class in problem1. So you can reuse and overwrite the code in problem 1.
    '''

    #--------------------------
    @staticmethod
    def cutting_points(X,Y):
        '''
            Find all possible cutting points in the continous attribute of X. 
            (1) sort unique attribute values in X, like, x1, x2, ..., xn
            (2) consider splitting points of form (xi + x(i+1))/2 
            (3) only consider splitting between instances of different classes
            (4) if there is no candidate cutting point above, use -inf as a default cutting point.
            Input:
                X: a list of values, a numpy array of int/float values.
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                cp: the list of  potential cutting points, a float numpy vector. 
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        cp = np.array([])
        sortedX = sorted(list(set(X)))
        if len(sortedX) == 1 or len(sorted(list(set(Y)))) == 1:
            cp = np.append(cp, float('-inf'))
            return cp
        cp0 = [(sortedX[i-1],sortedX[i])  for i in range(1,len(sortedX))]
        for item in cp0:
            l1, l2 = [], []
            for i in range(len(X)):
                if X[i] == item[0]:
                    l1.append(Y[i])
                if X[i] == item[1]:
                    l2.append(Y[i])
            if set(l1) != set(l2) or (len(set(l1)) > 1 and len(set(l2)) > 1):
                cp = np.append(cp, (item[0]+item[1])/2)

        #########################################
        return cp
    
    #--------------------------
    @staticmethod
    def best_threshold(X,Y):
        '''
            Find the best threshold among all possible cutting points in the continous attribute of X. 
            Input:
                X: a list of values, a numpy array of int/float values.
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                th: the best threhold, a float scalar. 
                g: the information gain by using the best threhold, a float scalar. 
            Hint: you can reuse your code in problem 1.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        cp = DT.cutting_points(X,Y)
        th = cp[0]
        g,cur_g = -1,float('-inf')
        sortedZ = sorted(zip(X,Y), key = lambda x:x[0])
        sortedX = [x[0] for x in sortedZ]
        sortedY = [x[1] for x in sortedZ]
        for i in cp:
            for j in range(1,len(sortedX)):
                if sortedX[j-1] < i and i < sortedX[j]:
                    cur_g = Tree.entropy(sortedY) - (len(sortedY[:j])/len(sortedY)*Tree.entropy(sortedY[:j]) + len(sortedY[j:])/len(sortedY)*Tree.entropy(sortedY[j:]))
                    break
            if cur_g > g:
                g = cur_g
                th = i

        #########################################
        return th,g 
    
    
    #--------------------------
    def best_attribute(self,X,Y):
        '''
            Find the best attribute to split the node. The attributes have continous values (int/float).
            Here we use information gain to evaluate the attributes. 
            If there is a tie in the best attributes, select the one with the smallest index.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
            Output:
                i: the index of the attribute to split, an integer scalar
                th: the threshold of the attribute to split, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        g = float('-inf')
        for k in range(X.shape[0]):
            cur_th,cur_g = DT.best_threshold(X[k,:],Y)
            if cur_g > g:
                g = cur_g
                th = cur_th
                i = k

        #########################################
        return i, th
    


        
    #--------------------------
    @staticmethod
    def split(X,Y,i,th):
        '''
            Split the node based upon the i-th attribute and its threshold.
            (1) split the matrix X based upon the values in i-th attribute and threshold
            (2) split the labels Y 
            (3) build children nodes by assigning a submatrix of X and Y to each node
    
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                i: the index of the attribute to split, an integer scalar
            Output:
                C1: the child node for values smaller than threshold
                C2: the child node for values larger than (or equal to) threshold
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        X1, X2 = [],[]
        Y1, Y2 = [],[]
        for k in range(X.shape[1]):
            if X[i,k] < th:
                X1.append(X[:,k])
                Y1.append(Y[k])
            if X[i,k] > th:
                X2.append(X[:,k])
                Y2.append(Y[k])
        X1, X2 = np.array(X1).T, np.array(X2).T
        Y1, Y2 = np.array(Y1), np.array(Y2)
        C1, C2 = Node(X1,Y1), Node(X2,Y2)

        #########################################
        return C1, C2
    
    
    
    #--------------------------
    def build_tree(self, t):
        '''
            Recursively build tree nodes.
            Input:
                t: a node of the decision tree, without the subtree built.
                t.X: the feature matrix, a numpy float matrix of shape n by p.
                   Each element can be int/float/string.
                    Here n is the number data instances, p is the number of attributes.
                t.Y: the class labels of the instances in the node, a numpy array of length n.
                t.C1: the child node for values smaller than threshold
                t.C2: the child node for values larger than (or equal to) threshold
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        t.p = Tree.most_common(t.Y)
        # if Condition 1 or 2 holds, stop recursion 
        if Tree.stop1(t.Y) or Tree.stop2(t.X):
            t.isleaf = True
            return
        # find the best attribute to split
        t.i, t.th = self.best_attribute(t.X,t.Y)
        t.C1, t.C2 = DT.split(t.X,t.Y,t.i,t.th)
        # recursively build subtree on each child node
        self.build_tree(t.C1)
        self.build_tree(t.C2)

        #########################################
    
    #--------------------------
    @staticmethod
    def inference(t,x):
        '''
            Given a decision tree and one data instance, infer the label of the instance recursively. 
            Input:
                t: the root of the tree.
                x: the attribute vector, a numpy vectr of shape p.
                   Each attribute value can be int/float
            Output:
                y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        # predict label, if the current node is a leaf node
        if t.isleaf:
            return t.p
        if x[t.i] < t.th:
            y = DT.inference(t.C1,x)
        else:
            y = DT.inference(t.C2,x)

        #########################################
        return y
    
    
    #--------------------------
    @staticmethod
    def predict(t,X):
        '''
            Given a decision tree and a dataset, predict the labels on the dataset. 
            Input:
                t: the root of the tree.
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float.
                   Here n is the number data instances in the dataset, p is the number of attributes.
            Output:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        Y = np.array([])
        for i in range(X.shape[1]):
            Y = np.append(Y,DT.inference(t,X[:,i]))

        #########################################
        return Y
    
    
    
    #--------------------------
    def train(self, X, Y):
        '''
            Given a training set, train a decision tree. 
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the training set, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
            Output:
                t: the root of the tree.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        t = Node(X,Y)
        self.build_tree(t) # What's the difference between using self and DT() ?

        #########################################
        return t


    #--------------------------
    @staticmethod
    def load_dataset(filename='data2.csv'):
        '''
            Load dataset 2 from the CSV file: data2.csv. 
            The first row of the file is the header (including the names of the attributes)
            In the remaining rows, each row represents one data instance.
            The first column of the file is the label to be predicted.
            In remaining columns, each column represents an attribute.
            Input:
                filename: the filename of the dataset, a string.
            Output:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element is a float scalar.
                   Here n is the number data instances in the dataset, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element is a string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        Z = np.loadtxt(filename, dtype=np.str, delimiter=',')
        X = Z[1:,1:].T.astype(float)
        Y = Z[1:,0]
        #########################################
        return X,Y




