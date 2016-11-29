# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 10:43:26 2016

@author: Kyle
"""

import pandas as pd
import scipy.sparse as sp
import numpy as np
import itertools as it
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from statistics import mean
from time import time, ctime
from math import ceil
import warnings
warnings.filterwarnings("ignore")

data_path = "~/Desktop/"
add_path = "C:\\Users\\Kyle\\OneDrive\\Documents\\GMU Classes\\CS 584\\HW4_Jackson_Truong\\data\\"

#reads in all the data as matrices, we only use test, train, and movie_tag, though
test_array = pd.read_table(add_path+"test.dat", skip_blank_lines=False, \
                     delim_whitespace=True).as_matrix()
train_array = pd.read_table(add_path+"train.dat", skip_blank_lines=False, \
                      delim_whitespace=True).as_matrix()
genre_array = pd.read_table(add_path+"movie_genres.dat", skip_blank_lines=False, \
                      delim_whitespace=True).as_matrix()
movie_tag_array = pd.read_table(add_path+"movie_tags.dat", \
                           skip_blank_lines=False).as_matrix()
actor_array = pd.read_table(add_path+"movie_actors.dat", \
                           skip_blank_lines=False).as_matrix()
actor_array = np.delete(actor_array, 2, 1)
director_array = pd.read_table(add_path+"movie_directors.dat", \
                           skip_blank_lines=False).as_matrix()[:,0:2]


#%%

#this section forms dicts to reindex the various IDs, to reduce dimensionality

users = np.unique(np.concatenate((train_array[:,0], test_array[:,0])))
users = dict(zip(users, np.arange(users.size))) 

#this mess is just so that every possible movie id is include
movies = np.unique(np.concatenate((train_array[:,1], test_array[:,1], \
                                  np.asarray(genre_array[:,0], dtype=int), \
                                  movie_tag_array[:,1], \
                                  np.asarray(director_array[:,0], dtype=int), \
                                  np.asarray(actor_array[:,0], dtype=int),)))

movies = dict(zip(movies, np.arange(movies.size)))

genres = np.unique(genre_array[:,1])
genres = dict(zip(genres, np.arange(genres.size)))

actors = np.unique(actor_array[:,1])
actors = dict(zip(actors, np.arange(actors.size)))
directors = np.unique(director_array[:,1])
directors = dict(zip(directors, np.arange(directors.size)))



#%%

#converts records in tabular form to sparse matrices

def dense2sparse(records, dict1=None, dict2=None, boolean=False):
    """ 
    Accepts a table, translating dictionaries and returns a sparse CSR matrix
    """
    # rating
    data = []
    # row, user
    i = []
    # column, movie
    j = []

    # iterate and tokenize each line
    for record in records:
        #if dictionary sent, use it to translate values
        #otherwise use the value raw
        if dict1==None:
            i.append(record[0])
        else:
            i.append(dict1[record[0]])
        if dict2==None:
            j.append(record[1])
        else:
            j.append(dict2[record[1]])
        #if boolean falso, assume third column exists with data values
        #otherwise just fill with ones
        if boolean==False:
            data.append(record[2])
        else:
            data.append(1)
    return sp.coo_matrix((data, (i,j)), shape = [np.amax(i)+1, np.amax(j)+1]).tocsr()
    
#%%
    
# conversion to sparse matrices
train_array = dense2sparse(train_array, users, movies)
genre_array = dense2sparse(genre_array, movies, genres, boolean=True)
movie_tag_array = dense2sparse(movie_tag_array, movies, None)
actor_array = dense2sparse(actor_array, movies, actors)
director_array = dense2sparse(director_array, movies, directors, boolean=True)
tfidf = TfidfTransformer()
movie_tag_array = tfidf.fit_transform(movie_tag_array)


#%%

def kNN(sparse_train, test_array, sparse_tags, k=20):
    """
    Makes movie rating predictions based on the train and test files
    ---------------------
    k: int
    Number of nearest numbers used for calculating movie prediction
    
    sparse_train: CSR matrix
    Format is row:userID and column:movieID
    
    test_array: numpy array
    Format is userID/movieID
    
    sparse_tags: CSR matrix
    Format is 
    """
    # output list
        
    test_y = []

    #perform cosine similarities outside loop for efficiency

    user_cos_similarity = cosine_similarity(sparse_train, sparse_train,\
                                            dense_output=False)   
    movie_cos_similarity = cosine_similarity(sparse_train.transpose(), \
                                             sparse_train.transpose(), \
                                            dense_output=False)
    tag_cos_similarity = cosine_similarity(sparse_tags, sparse_tags, \
                                            dense_output=False)

    # iterate through all lines in the test reviews and classify them
    for index, line in enumerate(test_array):
        # cosine similarity

        user = users[line[0]]
        movie = movies[line[1]]
        # get the indices of nearest users based on k parameter        
        user_neighbor_indices = user_cos_similarity[user].todense()
        user_neighbor_indices = np.array(user_neighbor_indices).flatten()
        user_neighbor_indices = user_neighbor_indices.argpartition(-k-1)[-k-1:]
        user_neighbor_indices = user_neighbor_indices[np.where(user_neighbor_indices != user)]
            # get list of ratings based on indices
        user_ratings = []
        for userID in user_neighbor_indices:
            user_ratings.append(sparse_train[userID, movie])
        
        # calculate rating for user
        user_ratings = np.array(user_ratings)
        user_ratings = user_ratings[np.nonzero(user_ratings)]

        if user_ratings.size == 0:
            # if no neighbors have a rating, take the average of all people
            #ratings = train_file[:,line[1]].data.tolist()
        
            # average rating for user and impute
            user_ratings = sparse_train.tocsc()[user,:].data
            # if there are no ratings for a given movie in the entire matrix,
            # assign a rating of 0 - failure
            try:            
                user_rating = user_ratings.mean()
            except:
                # assign rating of 0 if all else fails
                user_rating = 0
        else:
            user_rating = user_ratings.mean()
         
        #do same thing here but with comparing movies instead
        
        movie_neighbor_indices = movie_cos_similarity[movie].todense()
        movie_neighbor_indices = np.array(movie_neighbor_indices).flatten()
        movie_neighbor_indices = movie_neighbor_indices.argpartition(-k-1)[-k-1:]
        movie_neighbor_indices = movie_neighbor_indices[np.where(movie_neighbor_indices != movie)]
            # get list of ratings based on indices
        movie_ratings = []
        for movieID in movie_neighbor_indices:
            movie_ratings.append(sparse_train[user, movieID])
        
        # calculate rating for user
        movie_ratings = np.array(movie_ratings)
        movie_ratings = movie_ratings[np.nonzero(movie_ratings)]

        if movie_ratings.size == 0:
            # if no neighbors have a rating, take the average of all people
            #ratings = train_file[:,line[1]].data.tolist()
        
            # average rating for user and impute
            movie_ratings = sparse_train[:, movie].data #gets nonzero data
            movie_rating = movie_ratings.mean()
            
            if (np.isnan(movie_rating)): #if nan, means all values were 0
                # assign 0 if this happens
                movie_rating = 0
        else:
            movie_rating = movie_ratings.mean()
         
        #do same thing again but with tag data for comparing movies         
         
        tag_neighbor_indices = tag_cos_similarity[movie].todense()
        tag_neighbor_indices = np.array(tag_neighbor_indices).flatten()
        tag_neighbor_indices = tag_neighbor_indices.argpartition(-k-1)[-k-1:]
        tag_neighbor_indices = tag_neighbor_indices[np.where(tag_neighbor_indices != movie)]
      
        
        #get columns associated with nearest neighbors and get their averages
        #of nonzero entries. done en-masse here to speed up runtime
        new_ratings = sparse_train[:, tag_neighbor_indices]
        tag_ratings = np.true_divide(new_ratings.sum(0),(new_ratings!=0).sum(0))
            
        # remove nan values and return the mean of means
        tag_rating = np.nanmean(tag_ratings)

        #with three calculated means, if all are 0, return 3.5 (which is close)
        #to estimated global average. Otherwise take the average of valid
        #ratings given

        ratings = np.array([user_rating, movie_rating, tag_rating])
        
        ratings = ratings[np.nonzero(ratings)]
 
       
        if ratings.size == 0:
            new_rating = 3.5
        else:
            new_rating = np.mean(ratings)
            
        #sparse_train[user, movie] = new_rating
        test_y.append(new_rating)

        #test code to see if it returns nan
        if np.isnan(new_rating):
            print(index)
            print(ratings)
        
        #verbose code so that the user knows progress is being made
        if(index % 1000 == 0):
            print(ratings)
            print(index)
            print(ctime())
        
    return np.array(test_y)
        

#%%

submission = kNN(train_array, test_array, movie_tag_array, k=150)
np.savetxt("C:\\Users\\Kyle\\OneDrive\\Documents\\GMU Classes\\CS 584\\HW4_Jackson_Truong\\submissions\\submission" + "_" + str(ceil(time())) \
          + ".txt", submission, fmt='%s')

#%%
#validation testing
#this wasnt very good lol

orig_train_array = train_array

for _ in it.repeat(None, 5):

    train_array = orig_train_array
    np.random.shuffle(train_array)
    
    test_array = train_array[0:1001]
    train_array = train_array[1001:]
    
    sparse_train = dense2sparse(train_array, users)    
    submission = kNN(sparse_train, test_array[:, :2], 150, 0, 0)
    print(((submission-test_array[:,2]) ** 2).mean())

#%%

