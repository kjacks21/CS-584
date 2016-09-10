import pandas as pd
import numpy as np
import nltk
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
        # tokenize lines
        tokens = tokenizer.tokenize(line)
        # make tokens lowercase
        lower_tokens = [w.lower() for w in tokens if type(w) == str] 
        # remove numbers
        no_digits = [w for w in lower_tokens if not w.isdigit()]
        # remove stopwords
        stopped_tokens = [w for w in no_digits if not w in stopwords.words('english')]
        # stem words
        clean_tokens = [stemmer.stem(w) for w in stopped_tokens]
        # join text
        joined_text = " ".join(clean_tokens)
        # replace line with preprocessed line
        raw_text_df[index] = joined_text
        print(index)

preprocessing(train_X)
preprocessing(test_X)

# create tf-idf matrix for train and test
vectorizer = TfidfVectorizer()
train_tfidf = vectorizer.fit_transform(train_X)
test_tfidf = vectorizer.transform(test_X)

########################################################################  

def kNN(k, test_file, train_file):
    """
    implementation of k nearest neighbor algorithm using cosine similarity
    
    parameters
    ----------
        
    k: integer
    number of nearest neighbors used for classification
    
    test_tfidf: sparse matrix
    
    train_tfidf: sparse matrix
    
    """
    
    # iterate through all lines in the test reviews and classify them
    for index, line in enumerate(test_X):
        # cosine similarity
        cos_similarity = linear_kernel(test_file[index:index+1], train_file).flatten()
        # get the indices of nearest neighbors based on k parameter        
        neighbor_indices = cos_similarity.argsort()[:-k:-1]
        # get a list of labels from the neighbors and sum the list
        labels_list = train_y[neighbor_indices].tolist()
        label_sum = sum(labels_list)
        
        # if the label is 0 (tie between +1 and -1) use k-1 neighbors
        if label_sum == 0:
            kNN(k-1, test_file, train_file)
            
        # classify based on majority
        elif label_sum > 0:
            test_y.append("+1")
        else:
            test_y.append(-1)
            
        print(index)
            
    return pd.DataFrame(test_y)       

#########################################################################

test_y = []
submission = kNN(k=20, test_file=test_tfidf, train_file=train_tfidf)

# writing to .txt
np.savetxt(r'submissions/submission1.txt', submission.values, fmt='%s')

