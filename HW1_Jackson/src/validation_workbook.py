import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import os

os.chdir("C:/Users/Kyle/OneDrive/Documents/GMU Classes/CS 584/HW1_Jackson")

# reading in test and train
train = pd.read_table("data/train.data", header=None, skip_blank_lines=False)

# renaming column headers
train.columns = ["target", "data"]

# drop missing text labels in train to reduce noise
train = train.dropna()
train = train.reset_index(drop=True)

preprocessing(train.data)

# split text and target
X_train, X_test, y_train, y_test = cross_validation.train_test_split(train.data, train.target, test_size=0.2, random_state=0)

y_test = pd.DataFrame(y_test).reset_index(drop=True)


#########################################################################

# create tf-idf matrix for train and test
vectorizer = TfidfVectorizer()
train_tfidf = vectorizer.fit_transform(X_train)
test_tfidf = vectorizer.transform(X_test)

########################################################################  

# this kNN implementation is modified for cross_validation. See main_workbook.py
# for my kNN implementation
def kNN(k, testTfidf, trainTfidf, test_file, train_label, weight=True):
    test_y = []    
    # iterate through all lines in the test reviews and classify them
    for index, line in enumerate(test_file):
        # cosine similarity
        cos_similarity = linear_kernel(testTfidf[index:index+1], trainTfidf).flatten()
        
        
        if weight == True:
            # get the indices of nearest neighbors based on k parameter        
            neighbor_indices = cos_similarity.argsort()[:-k:-1]
            # similarities
            similarities = cos_similarity[neighbor_indices]
            # get a list of labels from the neighbors and sum the list
            labels_list = train_label[neighbor_indices].tolist()

            my_list = []            
            for s, l in zip(similarities, labels_list):
                if l == -1:
                    my_list.append(-s)
                else:
                    my_list.append(s)
            
            label_sum = sum(my_list)

            if label_sum > 0:
                test_y.append(1)
            else:
                test_y.append(-1)

        else:
            # get the indices of nearest neighbors based on k parameter        
            neighbor_indices = cos_similarity.argsort()[:-k:-1]
            # get a list of labels from the neighbors and sum the list
            labels_list = train_label[neighbor_indices].tolist()
            label_sum = sum(labels_list)

            if label_sum > 0:
                test_y.append(1)
            else:
                test_y.append(-1)
                
    return pd.DataFrame(test_y)   


#########################################################################

# code for validation step

# for loop iterates through possible values for k
k_list = []
weight_score_list = []
non_weight_score_list = []
for i in [5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41]:
    
    # weighted validation
    submission = kNN(k=i, testTfidf=test_tfidf, trainTfidf=train_tfidf, test_file=X_test, train_label=y_train, weight=True)    
    weight_df = pd.concat([submission, y_test], axis=1)
    weight_df.columns = ["pred","actual"]
    weight_df["score"] = 0
    weight_df["score"] = np.where(weight_df['pred'] == weight_df['actual'], 1, 0)         
        
    # not weighted validation
    submission1 = kNN(k=i, testTfidf=test_tfidf, trainTfidf=train_tfidf, test_file=X_test, train_label=y_train, weight=False)
    non_weight_df = pd.concat([submission1, y_test], axis=1)
    non_weight_df.columns = ["pred","actual"]
    non_weight_df["score"] = 0
    non_weight_df["score"] = np.where(non_weight_df['pred'] == non_weight_df['actual'], 1, 0)

    # compute score
    weight_score = sum(weight_df.score)/3700  
    non_weight_score = sum(non_weight_df.score)/3700
    
    print("With weight and k=", i, ", score is", weight_score)
    print("Without weight and k=", i, ", score is", non_weight_score)
    
    # create lists containing the scores for making a graphic    
    k_list.append(i)    
    weight_score_list.append(weight_score)
    non_weight_score_list.append(weight_score)

score_df = pd.DataFrame({"k":k_list, "weight_score":weight_score_list})

# validation scores plot
plt.scatter(score_df['k'], score_df['weight_score'])
plt.ylabel('Accuracy Score')
plt.xlabel('k-Neighbors')
plt.title('Validation Scores Using Varying k-Neighbor Values')
plt.show()


