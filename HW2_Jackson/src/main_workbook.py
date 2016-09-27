import pandas as pd
from sklearn import metrics
from sklearn import cross_validation
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from tpot import TPOTClassifier
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer
import numpy as np
import os
os.chdir("C:/Users/Kyle/OneDrive/Documents/GMU Classes/CS 584/HW2_Jackson")

# reading in test and train
train = pd.read_table("data/train.data", header=None, skip_blank_lines=False)
test = pd.read_table("data/test.data", header=None, skip_blank_lines=False)

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

# create boolean representations of the data
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
# TPOT building pipline
pipeline_optimizer = TPOTClassifier(generations=5, num_cv_folds=10, random_state=42, verbosity=2, scoring = "f1")
pipeline_optimizer.fit(new_train, label)
print(pipeline_optimizer.score(new_train, label))
pipeline_optimizer.export('tpot_exported_pipeline.py')


# classifier
exported_pipeline = make_pipeline(
    make_union(VotingClassifier([("est", ExtraTreesClassifier(criterion="gini", max_features=1.0, n_estimators=500))]), FunctionTransformer(lambda X: X)),
    make_union(VotingClassifier([("est", GradientBoostingClassifier(learning_rate=0.07, max_features=0.07, n_estimators=500))]), FunctionTransformer(lambda X: X)),
    BernoulliNB(alpha=0.08, binarize=0.66, fit_prior=True)
)

exported_pipeline.fit(new_train, label)
results = exported_pipeline.predict(new_test)

# cross-validation
scores_kbest = cross_validation.cross_val_score(exported_pipeline, new_train, label, cv=10, scoring='f1_weighted')

# writing submission
np.savetxt(r'submissions/submission5.txt', results, fmt='%s')
