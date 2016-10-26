# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 20:04:14 2016

@author: Kyle
"""


import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import linear_kernel
from random import randint
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# windows
words_df = pd.read_table("C:\\Users\\Kyle\\OneDrive\\Documents\\GMU Classes\\CS 584\\HW3_Jackson\\data\\words.txt", header=None, skip_blank_lines=False)
test = pd.read_table("C:\\Users\\Kyle\\OneDrive\\Documents\\GMU Classes\\CS 584\\HW3_Jackson\\data\\text_test.txt", header=None, skip_blank_lines=False)

test = test[0]

def dense2sparse(records):
    
    """ 
    Preprocesses text data for this assignment and returns a sparse matrix
    --------
    
    input_file: must contain the feature ids and counts for each sample in one line
    """

    data = []
    # row
    i = []
    # column
    j = []

    # iterate and tokenize each line
    for index, record in enumerate(records):
        
        features = map(int, record.split())
        
        feature_list = []
        count_list = []
        
        # create two seperate lists, one for the features and one for the counts
        for ind, rec in enumerate(features):
            if ind % 2 == 0:
                feature_list.append(rec)
            else:
                count_list.append(rec)
        
        for f, c in zip(feature_list, count_list):
            data.append(c)
            i.append(index)
            j.append(f)
    return sp.coo_matrix((data, (i,j)), shape = [records.size, 126373]).tocsr()

# conversion to sparse matrix
sparse_test = dense2sparse(test)

# conversion to tf-idf matrix
transformer = TfidfTransformer(norm="l2",smooth_idf=False)
tfidf_matrix = transformer.fit_transform(sparse_test)

# SVD aka LSA for dimesionality reduction
svd = TruncatedSVD(n_components=100, random_state=5)
reduced_tfidf = svd.fit_transform(tfidf_matrix)





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
        # if input_file is matrix: my_list.append(list(input_file[i].toarray()[0]))
        # if input_file is array: my_list.append(list(input_file[i]))
        my_list.append(list(input_file[i]))
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
        cos_similarity = linear_kernel(input_file[sample], centroid_dict[cluster]).flatten()
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
    for sample in cluster_dict:
        # nearest_centroid_dict structure {centroid : None (will become cosine sim)}
        nearest_centroid_dict = dict(a)
        # get all cos similarities for one sample to each centroid
        for centroid in nearest_centroid_dict:
            cos_similarity = linear_kernel(input_file[sample], centroid_dict[centroid]).flatten()
            nearest_centroid_dict[centroid] = cos_similarity[0]
        # assign sample to centroid with highest cos sim value
        cluster_dict[sample] = max(nearest_centroid_dict, key=nearest_centroid_dict.get)
    new_cohesion = cohesion(cluster_dict, centroid_dict, input_file)

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
                cos_similarity = linear_kernel(input_file[sample], centroid_dict[centroid]).flatten()
                nearest_centroid_dict[centroid] = cos_similarity[0]
            # assign sample to centroid with highest cos sim value
            cluster_dict[sample] = max(nearest_centroid_dict, key=nearest_centroid_dict.get)
        
        # recalculate centroid_dict
        for c in range(1,k+1):
            centroid_dict[c] = calculate_centroid(cluster_dict, input_file, c)
        
    
        previous_cohesion = new_cohesion
        new_cohesion = cohesion(cluster_dict, centroid_dict, input_file)
        print(new_cohesion)
        cohesions.append(new_cohesion)
        
        if previous_cohesion > new_cohesion:
            cluster_dict = dict(prev_cluster_dict)
        
    return cohesions, cluster_dict

########################################################################################

# for one submission
cohesions, cluster_dict = kMeans(reduced_tfidf)
submission = pd.DataFrame.from_dict(cluster_dict, orient='index')
np.savetxt("C:\\Users\\Kyle\\OneDrive\\Documents\\GMU Classes\\CS 584\\HW3_Jackson\\submissions\\submission_" + str(cohesions[-2]) + "_lsa6.txt", submission, fmt='%s')

# finding the best submission
# run 10 times
for i in range(11):
    cohesions, cluster_dict = kMeans(tfidf_matrix)
    submission = pd.DataFrame.from_dict(cluster_dict, orient='index')
    np.savetxt("C:\\Users\\Kyle\\OneDrive\\Documents\\GMU Classes\\CS 584\\HW3_Jackson\\submissions\\submission_" + str(cohesions[-2]) + ".txt", submission, fmt='%s')

# with lsa
for i in range(11):
    cohesions, cluster_dict = kMeans(reduced_tfidf)
    submission = pd.DataFrame.from_dict(cluster_dict, orient='index')
    np.savetxt("C:\\Users\\Kyle\\OneDrive\\Documents\\GMU Classes\\CS 584\\HW3_Jackson\\submissions\\submission_" + str(cohesions[-2]) + "_lsa.txt", submission, fmt='%s')

# calculate cohesion for differing k values
meta_dict = {}
for i in [3,5,7,9,11,13,15,17,19,21]:
    cohesions, cluster_dict = kMeans(tfidf_matrix, k=i)
    meta_dict[i] = cohesions[-2]
meta_cohesion = pd.DataFrame(list(meta_dict.items()))

# making plot
meta_cohesion.columns = ["centroids", "cohesion"]
data = meta_cohesion.sort_values(by="centroids")

######################################################################################

# with majority voting on the results






