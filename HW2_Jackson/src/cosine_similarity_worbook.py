# this workbook is used for identifying the clusters within the negative samples
# in the training set. It should be run after lines 1 to 70 are run in main_workbook.py
import statistics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import operator


negative_train = pd.DataFrame.copy(sparse_train_df)
positive_train = pd.DataFrame.copy(sparse_train_df)
positive_index = []
negative_index = []

# drop all samples that have a label of 1 (active) from negative_train
for index, label_ in enumerate(label):
    if label_ == 1:
        positive_index.append(index)
    else:
        negative_index.append(index)
        
negative_train.drop(negative_train.index[positive_index], inplace=True)
positive_train.drop(positive_train.index[negative_index], inplace=True)

# calculate cosine average similarity between one negative sample and all the positive samples
negative_averages = {}
for index, line in enumerate(negative_train):
    cosine_similarities = metrics.pairwise.cosine_similarity(negative_train[index:index+1], positive_train).flatten()
    negative_averages[index] = statistics.mean(cosine_similarities)
    print(index)

# get the top averages
top_negatives = dict(sorted(negative_averages.items(), key=operator.itemgetter(1), reverse=True)[:150])
tn_list = list(top_negatives.keys())

artificial_label = pd.DataFrame.copy(label)

# change most similar negatives to positives
for index, line in enumerate(artificial_label):
    if index in tn_list:
        line = 1
    print(index)



negative_train.drop(negative_train.index[positive_index], inplace=True)
positive_train.drop(positive_train.index[negative_index], inplace=True)

