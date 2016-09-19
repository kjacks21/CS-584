import pandas as pd
import numpy as np
import nltk
import re
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import os

os.chdir("C:/Users/Kyle/OneDrive/Documents/GMU Classes/CS 584/HW1_Jackson")

# reading in test and train
train = pd.read_table("data/train.data", header=None, skip_blank_lines=False)
test = pd.read_table("data/test.data", header=None, skip_blank_lines=False)

# renaming column headers
train.columns = ["target", "data"]
test.columns = ["data"]

# drop missing text labels in train to reduce noise
train = train.dropna()
train = train.reset_index(drop=True)
test.fillna("none", inplace=True)

# split text and target
train_X = train.data
train_y = train.target
test_X = test.data

#########################################################################

def preprocessing(raw_text_df):
    """ 
    preprocesses a DataFrame vector containing text
    """
    
    stemmer = nltk.stem.porter.PorterStemmer()
    tokenizer = RegexpTokenizer(r'\w+')
    # iterate over all lines for preprocessing
    for index, line in enumerate(raw_text_df):
        
        # if there is mention of stars from 1-5, change the integer into
        # text and combine the number and the word "star" to make a new word
        # example: "I give this product 1 star" is now "I give this product onestar"
        # why? numbers are removed as part of preprocessing
        if "1 star" in line:
            line = line.replace("1 star", "onestar")
        if "1 stars" in line:
            line = line.replace("1 stars", "onestar")
        if "2 star" in line:
            line = line.replace("2 star", "twostars")
        if "2 stars" in line:
            line = line.replace("2 stars", "twostars")
        if "3 star" in line:
            line = line.replace("3 star", "threestars")
        if "3 stars" in line:
            line = line.replace("3 stars", "threestars")
        if "4 star" in line:
            line = line.replace("4 star", "fourstars")
        if "4 stars" in line:
            line = line.replace("4 stars", "fourstars")
        if "5 star" in line:
            line = line.replace("5 star", "fivestars")
        if "5 stars" in line:
            line = line.replace("5 stars", "fivestars")
        
        # tokenize lines
        tokens = re.split('(\d+)',line)
        # remove numbers
        no_digits = [w for w in tokens if not w.isdigit()]
        # join tokens
        joined_text = " ".join(no_digits)
        # re tokenize
        tokens = tokenizer.tokenize(joined_text)
        # make tokens lowercase
        lower_tokens = [w.lower() for w in tokens if type(w) == str] 
        # remove stopwords
        stopped_tokens = [w for w in lower_tokens if not w in stopwords.words('english')]
        # stem words
        clean_tokens = [stemmer.stem(w) for w in stopped_tokens]
        # join text
        joined_text = " ".join(clean_tokens)
        # replace line with preprocessed line
        raw_text_df[index] = joined_text
        print(index)

# preprocessing raw_text
preprocessing(train_X)
preprocessing(test_X)


# read in already cleaned data
train_X = pd.read_table("data/clean_train.txt", header=None)
test_X = pd.read_table("data/clean_test.txt", header=None, skip_blank_lines=False)
train_X.columns = ["data"]
test_X.columns = ["data"]
train_X = train_X.data
test_X = test_X.data


# create tf-idf matrix for train and test
vectorizer = TfidfVectorizer()
train_tfidf = vectorizer.fit_transform(train_X)
test_tfidf = vectorizer.transform(test_X)

########################################################################  

def kNN(k, testTfidf, trainTfidf, test_file, train_label, weight=True):
    """
    implementation of k nearest neighbor algorithm using cosine similarity
    
    parameters
    ----------
        
    k: integer
    number of nearest neighbors used for classification
    
    testTfidf: sparse matrix
    
    trainTfidf: sparse matrix
    
    test_file: must be shape (int,)
    contains the preprocessed text for the kNN method to classify
    
    train_label: must be shape (int,)
    contains the labels from the training set
    
    weight: boolean
    default True. Uses simple weight to calculate classification
    
    """
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

            # make cosine similarity value negative or positive based on
            # its label and sum the cosine similarities
            my_list = []            
            for s, l in zip(similarities, labels_list):
                if l == -1:
                    my_list.append(-s)
                else:
                    my_list.append(s)   
                    
            label_sum = sum(my_list)
            #classify based on label_sum
            if label_sum > 0:
                test_y.append("+1")
            else:
                test_y.append(-1)

        else:
            # get the indices of nearest neighbors based on k parameter        
            neighbor_indices = cos_similarity.argsort()[:-k:-1]
            # get a list of labels from the neighbors and sum the list
            labels_list = train_label[neighbor_indices].tolist()
            label_sum = sum(labels_list)

            # classify based on label_sum
            if label_sum > 0:
                test_y.append("+1")
            else:
                test_y.append(-1)
                
        print(index)
            
    return pd.DataFrame(test_y)   

#########################################################################

submission = kNN(k=25, testTfidf=test_tfidf, trainTfidf=train_tfidf, test_file=test_X, train_label=train_y, weight=True)

# writing to .txt
np.savetxt(r'submissions/submission8-weight-k25-star.txt', submission.values, fmt='%s')

