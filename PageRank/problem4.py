import numpy as np
from problem3 import compute_P,random_walk

#-------------------------------------------------------------------------
'''
    Problem 4: Solving sink-node problem in PageRank
    In this problem, we implement the pagerank algorithm which can solve the sink node problem.
    You could test the correctness of your code by typing `nosetests test4.py` in the terminal.
'''

#--------------------------
def compute_S(A):
    '''
        compute the transition matrix S from addjacency matrix A, which solves sink node problem by filling the all-zero columns in A.
        S[j][i] represents the probability of moving from node i to node j.
        If node i is a sink node, S[j][i] = 1/n.
        Input: 
                A: adjacency matrix, a (n by n) numpy matrix of binary values. If there is a link from node i to node j, A[j][i] =1. Otherwise A[j][i]=0 if there is no link.
        Output: 
                S: transition matrix, a (n by n) numpy matrix of float values.  S[j][i] represents the probability of moving from node i to node j.
    The values in each column of matrix S should sum to 1.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    # S = A
    # sum_of_column = np.sum(S, axis=0)
    # for i in range(S.shape[1]):
    #     if sum_of_column[0, i] == 0:
    #         S[:,i] = 1.0/S.shape[1]
    #     else:
    #         for j in range(S.shape[0]):
    #             if S[j, i] == 1:
    #                 S[j, i] = 1.0/ sum_of_column[0, i]
    for i in range(A.shape[1]):
        if np.array_equal(A[:,i], np.zeros((A.shape[0], 1))):
            A[:,i] = np.ones((A.shape[0], 1))
        else:
            continue
    A_sum = A.sum(axis = 0)
    S = A / A_sum
    

    #########################################
    return S



#--------------------------
def pagerank_v2(A):
    ''' 
        A simplified version of PageRank algorithm, which solves the sink node problem.
        Given an adjacency matrix A, compute the pagerank score of all the nodes in the network. 
        Input: 
                A: adjacency matrix, a numpy matrix of binary values. If there is a link from node i to node j, A[j][i] =1. Otherwise A[j][i]=0 if there is no link.
        Output: 
                x: the ranking scores, a numpy vector of float values, such as np.array([[.3], [.5], [.7]])
    '''

    # Initialize the score vector with all one values
    num_nodes, _ = A.shape 
    x_0 =  np.asmatrix(np.ones((num_nodes,1))) 

    # compute the transition matrix from adjacency matrix
    S = compute_S(A)

    # random walk
    x, n_steps = random_walk(S,x_0)

    return x

