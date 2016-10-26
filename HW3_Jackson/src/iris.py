# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 18:22:05 2016

@author: Kyle
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from random import randint
from collections import defaultdict

test = pd.read_table("C:\\Users\\Kyle\\OneDrive\\Documents\\GMU Classes\\CS 584\\HW3_Jackson\\data\\iris_test.data", header=None, skip_blank_lines=False, delim_whitespace=True)

iris_array = pd.DataFrame.as_matrix(test)
iris_matrix = sp.csr_matrix(test.values)


def group_by_centroid(cluster_dict):
    """
    Given the cluster_dict, return a new dictionary with key, value structure
    {cluster : [sample indices]}
    """
    grouped_dict = defaultdict(list)
    for sample, cluster in cluster_dict.items():
        grouped_dict[cluster].append(sample)
    return dict(grouped_dict)

def get_arrays(grouped_dict, input_file, centroid):
    """
    Accepts grouped dictionary from group_by_centroid and returns a numpy array
    with all the arrays for a specific centroid
    """
    indices = grouped_dict[centroid]
    my_list = []
    for i in indices:
        my_list.append(list(input_file[i].toarray()[0]))
    return np.array(my_list)

def calculate_centroid(cluster_dict, input_file, centroid):
    """
    Accepts grouped_dict, return the mean numpy array for one centroid
    """
    grouped_dict = group_by_centroid(cluster_dict)
    arrays_ = get_arrays(grouped_dict, input_file, centroid)
    return np.mean(arrays_, axis=0)
    
def cohesion(cluster_dict, centroid_dict, input_file):
    
    """
    Calculates cohesion. Cohesion is equivelent to SSE, where we take the
    sum of all cosine similarities for samples and their assigned centroid
    """
    total_cohesion = 0
    for sample, cluster in cluster_dict.items():
        cos_similarity = cosine_similarity(input_file[sample], centroid_dict[cluster]).flatten()
        total_cohesion += cos_similarity
    # return cohesion as an integer
    return total_cohesion[0]
    

def kMeans(input_file, k=7):
    
    """
    implementation of kMeans clustering algorithm
    --------------------
    input_file: sparse matrix with [n_samples, n_features]
    
    k: int
    default=None. If None, kmeans will iterate through multiple k
    """
    
    # initial centroids
    n_samples = input_file.shape[0]
    
    # cluster_dict format is dictionary with:
    # {sample index : cluster assignment (e.g. 1)}
    print("Randomly assigning samples to clusters")
    cluster_dict = {}
    i = 0
    while i < n_samples:
        cluster_dict[i] = randint(1,k)
        i+=1
    print(cluster_dict)

    print("Initializing centroids")    
    # initial centroids
    # centroid_dict contains all centroids and is dictionary with:
    # {cluster (e.g. 1) : position array (mean of most similar samples)}
    centroid_dict = {}
    # nearest_centroid dictionary initialized here but not used until
    # assignment of sample to centroid
    a = {}
    for c in range(1,k+1):
        centroid_dict[c] = calculate_centroid(cluster_dict, input_file, c)
        a[c] = None
    # initial cohesion from random initialization
    initial_cohesion = cohesion(cluster_dict, centroid_dict, input_file)
    previous_cohesion = initial_cohesion    
    
    # first iteration of re-calculating centroids
    # iterate through all samples and calculate their cosine similarities to each
    # centroid
    print("First iteration of re-calculating centroids")
    print(cluster_dict)
    for key, value in cluster_dict.items():
        print(key, value)
        # nearest_centroid_dict structure {centroid : None (will become cosine sim)}
        nearest_centroid_dict = dict(a)
        # get all cos similarities for one sample to each centroid
        for centroid in nearest_centroid_dict:
            cos_similarity = cosine_similarity(input_file[key], centroid_dict[centroid]).flatten()
            nearest_centroid_dict[centroid] = cos_similarity[0]
        print(nearest_centroid_dict)
        # assign sample to centroid with highest cos sim value
        cluster_dict[key] = max(nearest_centroid_dict, key=nearest_centroid_dict.get)
        print(cluster_dict[key], max(nearest_centroid_dict, key=nearest_centroid_dict.get))
        
    new_cohesion = cohesion(cluster_dict, centroid_dict, input_file)
    print(cluster_dict)
    cohesions = [previous_cohesion, new_cohesion]    
    print(previous_cohesion)    
    
    # continue iterating until convergence, or cohesion does not increase
    print("Continuing until convergence.")
    while previous_cohesion < new_cohesion:
        prev_cluster_dict = dict(cluster_dict)
        for sample in cluster_dict:
            # nearest_centroid_dict structure {centroid : None (will become cosine sim)}
            nearest_centroid_dict = dict(a)
            # get all cos similarities for one sample to each centroid
            for centroid in nearest_centroid_dict:
                cos_similarity = cosine_similarity(input_file[sample], centroid_dict[centroid]).flatten()
                nearest_centroid_dict[centroid] = cos_similarity[0]
            # assign sample to centroid with highest cos sim value
            cluster_dict[sample] = max(nearest_centroid_dict, key=nearest_centroid_dict.get)
        
        # recalculate centroid_dict
        print(cluster_dict)        
        for c in range(1,k+1):
            centroid_dict[c] = calculate_centroid(cluster_dict, input_file, c)
        
    
        previous_cohesion = new_cohesion
        new_cohesion = cohesion(cluster_dict, centroid_dict, input_file)
        print(new_cohesion)
        cohesions.append(new_cohesion)
        
        if previous_cohesion > new_cohesion:
            cluster_dict = dict(prev_cluster_dict)
        
    return cohesions, cluster_dict


# for one submission

cohesions, cluster_dict = kMeans(iris_matrix, k=3)
submission = pd.DataFrame.from_dict(cluster_dict, orient='index')
np.savetxt("C:\\Users\\Kyle\\OneDrive\\Documents\\GMU Classes\\CS 584\\HW3_Jackson\\submissions\\iris_submission_" + str(cohesions[-2]) + ".txt", submission, fmt='%s')
