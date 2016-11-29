# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 10:43:26 2016

@author: Kyle
"""

import pandas as pd
import scipy.sparse as sp
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from statistics import mean

data_path = "C:\\Users\\Kyle\\OneDrive\\Documents\\GMU Classes\\CS 584\\HW4_Jackson_Truong\\data\\"
test = pd.read_table(data_path+"test.txt", header=None, skip_blank_lines=False, delim_whitespace=True)
train = pd.read_table(data_path+"train.txt", header=None, skip_blank_lines=False)

train_ = train[0]
# numpy array with format: [userID, movieID]
test_array = pd.DataFrame.as_matrix(test)

def dense2sparse(records):
    """ 
    Accepts raw movie rating train data and returns a sparse CSR matrix
    --------
    input_file: userID, movieID, rating
    """
    # rating
    data = []
    # row, user
    i = []
    # column, movie
    j = []

    # iterate and tokenize each line
    for record in records:
        features = list(record.split())
        i.append(int(features[0]))
        j.append(int(features[1]))
        data.append(float(features[2]))

    return sp.coo_matrix((data, (i,j)), shape = [records.size, 100000]).tocsr()

# conversion to sparse matrix
sparse_train = dense2sparse(train_)


def NN(train_file, test_file, k=20):
    """
    Makes movie rating predictions based on the train and test files
    ---------------------
    n_neighbors: int
    Number of nearest numbers used for calculating movie prediction
    
    train: sparse CSR matrix
    Format is userID/movieID/rating
    
    test: numpy array
    Format is userID/movieID
    """
    # output list
    test_y = []    
    
    # iterate through all lines in the test reviews and classify them
    for index, line in enumerate(test_file):
        # cosine similarity
        cos_similarity = cosine_similarity(train_file[line[0],:], train_file).flatten()
        # get the indices of nearest neighbors based on k parameter        
        neighbor_indices = cos_similarity.argsort()[:-(k+1):-1].tolist()
        neighbor_indices = neighbor_indices[1:]
            # get list of ratings based on indices
        ratings = []
        for userID in neighbor_indices:
            ratings.append(train_file[userID, line[1]])
        
        # calculate rating for user
        ratings = np.asarray(ratings)
        ratings[ratings == 0] = np.nan

        if all(np.isnan(ratings)):
            # if no neighbors have a rating, take the average of all people
            #ratings = train_file[:,line[1]].data.tolist()
        
            # average rating for user and impute
            ratings = train_file[line[0],:].data.tolist()
            # when there are no ratings for a given movie in the entire matrix,
            # assign a rating of 3.0
            try:            
                rating = mean(ratings)
            except:
                try:
                    # get average rating for movie
                    ratings = train_file[:,line[1]].data.tolist()
                    rating = mean(ratings)
                except:
                    # assign rating of 3 if all else fails
                    rating = 3.0
        else:
            rating = np.nanmean(ratings)
        test_y.append(rating)
        
        if index == 71298:
            print(index)
        
    return pd.DataFrame(test_y)


submission = NN(sparse_train, test_array, k=150)
np.savetxt(r'C:\\Users\\Kyle\\OneDrive\\Documents\\GMU Classes\\CS 584\\HW4_Jackson_Truong\\submissions\\submission_150_', submission.values, fmt='%s')

