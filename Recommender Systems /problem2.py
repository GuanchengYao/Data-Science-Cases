import math
import numpy as np
#-------------------------------------------------------------------------
'''
    Problem 2: User-based recommender systems
    In this problem, you will implement a version of the recommender system using user-based method.
    You could test the correctness of your code by typing `nosetests test2.py` in the terminal.
'''

#--------------------------
def cosine_similarity(RA, RB):
    '''
        compute the cosine similarity between user A and user B. 
        The similarity values between users are measured by observing all the items which have been rated by BOTH users. 
        If an item is only rated by one user, the item will not be involved in the similarity computation. 
        You need to first remove all the items that are not rated by both users from RA and RB. 
        If the two users don't share any item in their ratings, return 0. as the similarity.
        Then the cosine similarity is < RA, RB> / (|RA|* |RB|). 
        Here <RA, RB> denotes the dot product of the two vectors (see here https://en.wikipedia.org/wiki/Dot_product). 
        |RA| denotes the L-2 norm of the vector RA (see here for example: http://mathworld.wolfram.com/L2-Norm.html). 
        For more details, see here https://en.wikipedia.org/wiki/Cosine_similarity.
        Input:
            RA: the ratings of user A, a float python vector of length m (the number of movies). 
                If the rating is unknown, the number is 0. For example the vector can be like [0., 0., 2.0, 3.0, 0., 5.0]
            RB: the ratings of user B, a float python vector
                If the rating is unknown, the number is 0. For example the vector can be like [0., 0., 2.0, 3.0, 0., 5.0]
        Output:
            S: the cosine similarity between users A and B, a float scalar value between -1 and 1.
        Hint: you could use math.sqrt() to compute the square root of a number
    '''
    #########################################
    ## INSERT YOUR CODE HERE

    # assuming cosine(RA, RB) = P/(sqrt(NA)*sqrt(NB))
    P, A, B = 0, 0, 0
    # loop through two lists
    for i in range(len(RA)):
        P = P + RA[i] * RB[i]
        # if both users rated the item 
        if RA[i]!= 0 and RB[i]!= 0:
            A = A + RA[i]**2
            B = B + RB[i]**2   
    # if the two user share no item in their ratings
    if A == 0 and B == 0:
        return 0
    # compute cosine similarity on the shared items 
    S = P / (math.sqrt(A) * math.sqrt(B))  

    #########################################
    return S 


#--------------------------
def find_users(R, i):
    '''
        find the all users who have rated the i-th movie.  
        Input:
            R: the rating matrix, a float numpy matrix of shape m by n. Here m is the number of movies, n is the number of users.
                If a rating is unknown, the number is 0. 
            i: the index of the i-th movie, an integer python scalar (Note: the index starts from 0)
        Output:
            idx: the indices of the users, a python list of integer values 
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    idx = []
    for j in range(R.shape[1]):
        if R[i][j] != 0:
            idx.append(j)
    #########################################
    return idx

#--------------------------
def user_similarity(R, j, idx):
    '''
        compute the cosine similarity between a collection of users in idx list and the j-th user.  
        Input:
            R: the rating matrix, a float numpy matrix of shape m by n. Here m is the number of movies, n is the number of users.
                If a rating is unknown, the number is 0. 
            j: the index of the j-th user, an integer python scalar (Note: the index starts from 0)
            idx: a list of user indices, a python list of integer values 
        Output:
            sim: the similarity between any user in idx list and user j, a python list of float values. It has the same length as idx.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    sim = []
    for i in range(R.shape[1]):
        if i in idx:
            sim.append(float(cosine_similarity(R[:,i], R[:,j])))

    #########################################
    return sim 


#--------------------------
def user_based_prediction(R, i_movie, j_user, K=5):
    '''
        Compute a prediction of the rating of the j-th user on the i-th movie using user-based approach.  
        First we take all the users who have rated the i-th movie, and compute their similarities to the target user j. 
        If there is no user who has rated the i-th movie, predict 3.0 as the default rating.
        From these users, we pick top K similar users. 
        If there are less than K users who has rated the i-th movie, use all these users.
        We weight the user's ratings on i-th movie by the similarity between that user and the target user. 
        Finally, we rescale the prediction by the sum of similarities to get a reasonable value for the predicted rating.
        Input:
            R: the rating matrix, a float numpy matrix of shape m by n. Here m is the number of movies, n is the number of users.
                If the rating is unknown, the number is 0. 
            i_movie: the index of the i-th movie, an integer python scalar
            j_user: the index of the j-th user, an integer python scalar
            K: the number of similar users to compute the weighted average rating.
        Output:
            p: the predicted rating of user j on movie i, a float scalar value between 1. and 5.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    p = 0
    # find all other users who have rated movie i.
    idx = find_users(R, i_movie)
    if len(idx) == 0:
        p = 3.0
        return p
    # if there are less than K users who have rated the movie, change K to the number of users
    if K > len(idx):
        K = len(idx)
    # compute the similarity between all of these users with user j
    sim = user_similarity(R, j_user, idx)
    # compute the weighted average of the top K similar users to user j
    values = []
    for i in range(len(idx)):
        values.append(R[i_movie,idx[i]])

    for i in range(len(sim)):
        max_index = i
        for j in range(i+1, len(sim)):
            if sim[j] > sim[max_index]:
                max_index = j
        sim[i], sim[max_index] = sim[max_index], sim[i]
        values[i], values[max_index] = values[max_index], values[i]
    for i in range(K):
        p += sim[i]*values[i]
    p = p/sum(sim[:K])
    #p = (sum(sim[:K])/K) * (sum(values[:K])/K)
    #########################################
    return p 


#--------------------------
def compute_RMSE(ratings_pred, ratings_real):
    '''
        Compute the root of mean square error of the rating prediction.
        Input:
            ratings_pred: predicted ratings, a float python list
            ratings_real: real ratings, a float python list
        Output:
            RMSE: the root of mean squared error of the predicted rating, a float scalar.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    sum = 0
    for i in range(len(ratings_pred)):
        sum += (ratings_pred[i] - ratings_real[i]) ** 2
    RMSE = math.sqrt(sum / len(ratings_pred))
    #########################################
    return RMSE



#--------------------------
def load_rating_matrix(filename = 'movielens_train.csv'):
    '''
        Load the rating matrix from a CSV file.  In the CSV file, each line represents (user id, movie id, rating).
        Note the ids start from 1 in this dataset.
        Input:
            filename: the file name of a CSV file, a string
        Output:
            R: the rating matrix, a float numpy array of shape m by n. Here m is the number of movies, n is the number of users.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    R = []
    data = np.array(np.loadtxt(filename, dtype=np.int, delimiter=','))
    m = max(data[:,1])
    n = max(data[:,0])
    R = np.zeros((m,n))
    tem = []
    for j in range(len(data)):
        tem = data[j,:]
        R[tem[1]-1,tem[0]-1] = tem[2]

    #########################################
    return R


#--------------------------
def load_test_data(filename = 'movielens_test.csv'):
    '''
        Load the test data from a CSV file.  In the CSV file, each line represents (user id, movie id, rating).
        Note the ids in the CSV file start from 1. But the indices in u_ids and m_ids start from 0.
        Input:
            filename: the file name of a CSV file, a string
        Output:
            m_ids: the list of movie ids, an integer python list of length n. Here n is the number of lines in the test file. (Note indice should start from 0)
            u_ids: the list of user ids, an integer python list of length n. 
            ratings: the list of ratings, a float python list of length n. 
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    data = np.array(np.loadtxt(filename, dtype=np.int, delimiter=','))
    m_ids = data[:,1].tolist()
    m_ids = list(map(lambda x: x-1, m_ids))
    u_ids = data[:,0].tolist()
    u_ids = list(map(lambda x: x-1, u_ids))
    ratings = data[:,2].astype(float).tolist()


    #########################################
    return m_ids, u_ids, ratings


#--------------------------
def movielens_user_based(train_file='movielens_train.csv', test_file ='movielens_test.csv', K = 5):
    '''
        Compute movie ratings in movielens dataset. Based upon the training ratings, predict all values in test pairs (movie-user pair).
        In the training file, each line represents (user id, movie id, rating).
        Note the ids start from 1 in this dataset.
        Input:
            train_file: the train file of the dataset, a string.
            test_file: the test file of the dataset, a string.
            K: the number of similar users to compute the weighted average rating.
        Output:
            RMSE: the root of mean squared error of the predicted rating, a float scalar.
    Note: this function may take 1-5 minutes to run.
    '''
   
    # load training set
    R = load_rating_matrix(train_file)
    # load test set
    m_ids, u_ids,ratings_real = load_test_data(test_file)
    # predict on test set
    #########################################
    ## INSERT YOUR CODE HERE
    ratings_pred = []
    for i in range(len(m_ids)):
        ratings_pred.append(user_based_prediction(R, m_ids[i], u_ids[i], K))

    #########################################
    # compute RMSE 
    RMSE = compute_RMSE(ratings_pred,ratings_real)
    return  RMSE 


