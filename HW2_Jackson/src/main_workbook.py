import pandas as pd
from sklearn import metrics
from sklearn import cross_validation
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from tpot import TPOTClassifier
import numpy as np
import os
os.chdir("C:/Users/Kyle/OneDrive/Documents/GMU Classes/CS 584/HW2_Jackson/src")

# reading in test and train
train = pd.read_table("C:/Users/Kyle/OneDrive/Documents/GMU Classes/CS 584/HW2_Jackson/data/train.data", header=None, skip_blank_lines=False)
test = pd.read_table("C:/Users/Kyle/OneDrive/Documents/GMU Classes/CS 584/HW2_Jackson/data/test.data", header=None, skip_blank_lines=False)

label = train[0]
train = train[1]
test = test[0]

#####################################################################
# data munging

# returns sparse DataFrame
def dense2sparse(train_df, test_df, is_test=False):
    # get dictionary of features from entire file
    features_list = []
    for line in train_df:
        tokens = line.split()
        features_list.extend(tokens)
    for line in test_df:
        tokens = line.split()
        features_list.extend(tokens)


    if is_test == True:
        meta_list = []
        # creating sparse dictionary with feature counts based on line
        for index, line in enumerate(test_df):
            # line_features is the dict that will be converted to a pandas DataFrame
            features = dict.fromkeys(set(features_list),0)
            tokens = line.split()
            line_dict = Counter(tokens)
            # comparing the Counter dictionary and the {feature:0} dictionary
            # and inputing counts from line_dict to line_features
            for key, value in line_dict.items():
                if key in features:
                    features[key] = value
            
            # append dictionaries of each line to a meta list to be transormed
            # to a DataFrame
            meta_list.append(features)
    else:
        meta_list = []
        # creating sparse dictionary with feature counts based on line
        for index, line in enumerate(train_df):
            # line_features is the dict that will be converted to a pandas DataFrame
            features = dict.fromkeys(set(features_list),0)
            tokens = line.split()
            line_dict = Counter(tokens)
            # comparing the Counter dictionary and the {feature:0} dictionary
            # and inputing counts from line_dict to line_features
            for key, value in line_dict.items():
                if key in features:
                    features[key] = value
            
            # append dictionaries of each line to a meta list to be transormed
            # to a DataFrame
            meta_list.append(features)
        
    return pd.DataFrame(meta_list)

sparse_train_df = dense2sparse(train, test)
sparse_test_df = dense2sparse(train, test, is_test=True)


######################################################################
# feature engineering
sparse_train_df.shape
# selecting the top features
kbest = SelectKBest(chi2, k=1000)
new_train = kbest.fit_transform(sparse_train_df, label)
new_test = kbest.transform(sparse_test_df)



#######################################################################
# TPOT
my_tpot = TPOTClassifier(generations=10)  
my_tpot = my_tpot.fit(sparse_train_df, label)
tpot_output = my_tpot.predict(sparse_test_df)


#######################################################################
# cross validation

rf = RandomForestClassifier(n_estimators=1000, random_state=1, class_weight="balanced")

scores = cross_validation.cross_val_score(rf, sparse_train_df, label, cv=10, scoring='f1_weighted')

scores_kbest = cross_validation.cross_val_score(rf, new_train, label, cv=10, scoring='f1_weighted')



########################################################################


rf = rf.fit(new_train, label)

output = rf.predict(new_test)

np.savetxt(r'C:/Users/Kyle/OneDrive/Documents/GMU Classes/CS 584/HW2_Jackson/submissions/submission5.txt', output, fmt='%s')
