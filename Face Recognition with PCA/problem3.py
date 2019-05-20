import numpy as np
import math
# Hint: you can reuse the functions that you have implemented in problem 3. For example, p3.reshape_to_image() 
import problem2 as p2

#-------------------------------------------------------------------------
'''
    Problem 4: face recognition 
    In this problem, you will use PCA to perform face recogition in a face image dataset.
    We assume you have already passed the unit tests in problem3.py. There will be a file named "face_pca.npy" in your folder.
    This file contains a numpy matrix of shape (400,20), which is the reduced dimsions for the 400 face images.
    We will need to use this file in this problem.
    You could test the correctness of your code by typing `nosetests test4.py` in the terminal.

    Notations:
            ---------- input data ------------------------
            n: the number of face images, an integer scalar. (n = 400)
            p: the number of pixels in each image, an integer scalar. (p= 4096)
            c: the number of classes (persons), an integer scalar. (c = 40)
            k: the number of dimensions to reduce to, an integer scalar.
            Xp: the feature matrix with reduced dimensions, a numpy float matrix of shape n by k. (400 by k) 
            ----------------------------------------------
'''

#--------------------------
def compute_distance(Xp, q):
    '''
        Compute the Euclidean distance between an query image and all the images in an image dataset.  
        Intput:
            Xp: the projected feature matrix, a float numpy matrix of shape (400, k). 
            q:  a projected features of a query face image. a numpy vector of shape (1, k). 
        Output:
            d: distances between the query image and all the images in Xp. A numpy vector of shape (400,1), where each element i, is the Euclidean distance between i-th image in X and the query image.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    d = []
    for i in range(Xp.shape[0]):
        sum_d = 0
        for j in range(len(q)):
            sum_d = sum_d + (Xp[i,j]-q[j]) ** 2
        d.append(math.sqrt(sum_d))
    d = np.array(d)

    #########################################
    return d


#--------------------------
def face_recognition(Xp,q):
    '''
        Compute the most similar faces to the query face (id) from all the images in an image dataset.  
        We will use one image from olivetti face dataset as the query and search for similar faces to the query.
        Intput:
            Xp: the projected feature matrix, a float numpy matrix of shape (400, k). 
            q:  a projected features of a query face image. a numpy vector of shape (1, k). 
        Output:
            ids: the ranked ids of similar face images to the query image.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    d = compute_distance(Xp, q)
    keys = []
    values = []
    for idx, val in enumerate(d):
        keys.append(idx)
        values.append(val)
    z = dict(zip(keys,values))
    sorted_d = sorted(z.items(), key = lambda x:x[1])
    ids = np.array(list([i[0] for i in sorted_d]))

    #########################################
    return ids




