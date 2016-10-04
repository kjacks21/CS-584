# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 11:07:58 2016

@author: Robert
"""

import pandas as pd
from time import time
from math import ceil
import scipy.sparse as sp
from sklearn import cross_validation
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import BernoulliNB
import imblearn.under_sampling as us

import numpy as np


# reading in test and train

#Windows
train = pd.read_table("C:\\Users\\Kyle\\OneDrive\\Documents\\GMU Classes\\CS 584\\HW2_Jackson_Truong\\src", header=None, skip_blank_lines=False)
test = pd.read_table("C:\\Users\\Kyle\\OneDrive\\Documents\\GMU Classes\\CS 584\\HW2_Jackson_Truong\\src", header=None, skip_blank_lines=False)


#Linux
#train = pd.read_table("/home/robert/Desktop/drugs_train.data", header=None, skip_blank_lines=False)
#test = pd.read_table("/home/robert/Desktop/drugs_test.data", header=None, skip_blank_lines=False)

label = np.array(train[0])

train = train[1]
test = test[0]

#%%
# data munging

# returns sparse scipy matrix
def dense2sparse(records):

    data = []
    i = []
    j = []

    for index, record in enumerate(records):
        features = map(int, record.split())
        for feature in features:
            data.append(1)
            i.append(index)
            j.append(feature - 1)
    return sp.coo_matrix((data, (i,j)), shape = [records.size, 100000]).tocsr()

#%%

# create sparse representations of the data|
sparse_train = dense2sparse(train)
sparse_test = dense2sparse(test)

#%%

# feature engineering
# selecting the top features

k_features = 225
kbest = SelectKBest(score_func=chi2, k=k_features)
reduced_train = kbest.fit_transform(sparse_train.toarray(), label)
reduced_test = kbest.transform(sparse_test.toarray())

#Use undersampling algorithm

enn = us.EditedNearestNeighbours(random_state=ceil(time()))
train_res, label_res = enn.fit_sample(reduced_train, label)
train_res = sp.csr_matrix(train_res)

# Bernoulli NB

bnb = BernoulliNB(alpha=.75)
bnb.fit(train_res, label_res)
results = bnb.predict(reduced_test)


# Shuffling data for cross validation

permutation = np.arange(label.size)
np.random.shuffle(permutation)

shuffled_train = reduced_train[permutation]
shuffled_label = label[permutation]

# cross-validation
scores_kbest = cross_validation.cross_val_score(bnb, shuffled_train, shuffled_label, cv=10, scoring='f1')

print(str(scores_kbest.mean(0)))

#%%

# writing submission
np.savetxt('C:\\Users\\Robert\\Desktop\\submission' + "_" + str(ceil(time())) + "_ " + str(k_features) + ".txt", results, fmt='%s')


#%%
# TPOT building pipline
#pipeline_optimizer = TPOTClassifier(generations=5, num_cv_folds=5, random_state=42, verbosity=2, scoring = "f1")
#pipeline_optimizer.fit(new_train, label)
#print(pipeline_optimizer.score(new_train, label))
#pipeline_optimizer.export('tpot_exported_pipeline' + "_" + str(ceil(time())) + "_ " + str(k_features) + ".py")

#%%

## classifier from TPOT output
#exported_pipeline = make_pipeline(
#    make_union(VotingClassifier([("est", ExtraTreesClassifier(criterion="gini", max_features=1.0, n_estimators=500))]), FunctionTransformer(lambda X: X)),
#    make_union(VotingClassifier([("est", GradientBoostingClassifier(learning_rate=0.07, max_features=0.07, n_estimators=500))]), FunctionTransformer(lambda X: X)),
#    BernoulliNB(alpha=0.08, binarize=0.66, fit_prior=True)
#)
#
#exported_pipeline.fit(new_train, label)
#results = exported_pipeline.predict(new_test)
