# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 19:21:28 2022

@author: Joyun

The goal of the project is to predict generes of each song with track and echo info
"""

###############################
### Import Libraries       ####
###############################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline


###############################
### Read in Files  ############
###############################

tracks = pd.read_csv('datasets/fma-rock-vs-hiphop.csv')
echo = pd.read_json('datasets/echonest-metrics.json', precise_float=True)
tracks = tracks[['track_id', 'genre_top']]

# Merge the relevant columns of tracks and echonest_metrics
echo_tracks = echo.merge(tracks, how = 'inner', on='track_id')

# Inspect the resultant dataframe
echo_tracks.info()


###################################
####### EDA and Preprocess ########
###################################

# check multicollinearity with pairwise correlation matrix.
# multicollinearity will cause severe problems with the following models:
# regression, logistics, knn, naive basyes model, etc
# Excessive features will also cause overfitting and longer computation time

corr_table = echo_tracks.corr()
corr_table.style.background_gradient(cmap='coolwarm')

# no pairwise feature is highly correlatted

###################################
####### Split Data         ########
###################################

# Create features
features = echo_tracks.drop(["genre_top", "track_id"], axis=1).values 

# Create labels
labels = echo_tracks["genre_top"].values


# Split our data
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, random_state=10)

######################################################
####### Dimensionality Reduction with PCA     ########
######################################################

# Performing Standardization Before PCA

# Scale train_features and test_features
scaler = StandardScaler()
scaled_train_features = scaler.fit_transform(train_features)
scaled_test_features = scaler.transform(test_features)

# Get our percentage of variance explained from PCA using all features
pca = PCA()
pca.fit(scaled_train_features)
exp_variance = pca.explained_variance_ratio_

# determining the number of components to use with scree plot
fig, ax = plt.subplots()
ax.bar(range(pca.n_components_), exp_variance)
ax.set_xlabel('# of Principal Component')
# the elbow is not obvious

# method 2
# Determining the number of components with cumulative explained variance plot
# Calculate the cumulative explained variance
cum_exp_variance = np.cumsum(exp_variance)

# Plot the cumulative explained variance and draw a dashed line at 0.90.
fig, ax = plt.subplots()
ax.plot(cum_exp_variance)
ax.axhline(y=0.90, linestyle='-', color ='red')
# 6 features is enough to explain 90% of the variance in genre

pca = PCA(n_components=6, random_state=10)
# Fit and transform the scaled training features using pca
train_pca = pca.fit_transform(scaled_train_features)

# transform the scaled test features using pca
test_pca = pca.transform(scaled_test_features)


###########################################
####### Build Decision Tree Model  ########
###########################################

# Create our decision tree
tree = DecisionTreeClassifier(random_state=10)

# Train our decision tree
tree.fit(train_pca, train_labels)

# Predict the labels for the test data
pred_labels_tree = tree.predict(test_pca)
print(pred_labels_tree)

###########################################
####### Build Logistics Regression ########
###########################################

# Train our logisitic regression
logreg = LogisticRegression(random_state=10)
logreg.fit(train_pca, train_labels)

# Predict the labels for the test data
pred_labels_logit = logreg.predict(test_pca)

###########################################
####### Comparison Between Models  ########
###########################################
report_tree = classification_report(test_labels, pred_labels_tree)
report_logistics = classification_report(test_labels, pred_labels_logit)
print('==================================================')
print("Decision Tree: \n", report_tree )
print('===================================================')
print("Logistic Regression: \n", report_logistics)

# the scores for hip hop is lower than that of Rock in both decision tree and 
# logistic regression due to imbalance dataset: there are more rock than hip hop songs


###############################################################################0

###########################################
#######          Balanced Data     ########
###########################################

hop_only = echo_tracks[echo_tracks['genre_top'] == 'Hip-Hop']
rock_only = echo_tracks[echo_tracks['genre_top'] == 'Rock']

# subset the rock songs to the same number of rows as the minority group (hip hop)
rock_only = rock_only.sample(hop_only.shape[0], random_state=10)

# stack the dataframes hop_only and rock_only
rock_hop_bal = pd.concat([rock_only, hop_only])


features_bal = rock_hop_bal.drop(['genre_top', 'track_id'], axis=1) 
labels_bal = rock_hop_bal['genre_top']

###########################################
#######      Split Train and Test    ######
###########################################
# Redefine the train and test set with the pca_projection from the balanced data
train_features_bal, test_features_bal, train_labels_bal, test_labels_bal = train_test_split(
    features_bal, labels_bal, random_state=10)

###########################################
#######    PCA with Balanced Data    ######
###########################################
train_pca_bal = pca.fit_transform(scaler.fit_transform(train_features_bal))
test_pca_bal = pca.transform(scaler.transform(test_features_bal))


###########################################
#######    Tree with Balanced Data    #####
###########################################

# Train our decision tree on the balanced data
tree = DecisionTreeClassifier(random_state=10)
tree.fit(train_pca_bal, train_labels_bal)
pred_labels_tree_bal = tree.predict(test_pca_bal)

################################################
#######    Logistics with Balanced Data    #####
################################################

# Train our logistic regression on the balanced data
logreg = LogisticRegression(random_state=10)
logreg.fit(train_pca_bal, train_labels_bal)
pred_labels_logit_bal = logreg.predict(test_pca_bal)

# compare the models
print('==================Balanced Dataset================================')
print("Decision Tree: \n", classification_report(test_labels_bal, pred_labels_tree_bal))
print('===================================================================')
print("Logistic Regression: \n", classification_report(test_labels_bal, pred_labels_logit_bal))

# it improved!
# we will use the balanced dataset to train our models instead of the original dataset

###########################################
#######    K fold cross validation    #####
###########################################

tree_pipe = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=6)), 
                      ("tree", DecisionTreeClassifier(random_state=10))])
logreg_pipe = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=6)), 
                        ("logreg", LogisticRegression(random_state=10))])

# Set up our K-fold cross-validation
kf = KFold(10)

# Train our models using KFold cv
tree_score = cross_val_score(tree_pipe, features_bal, labels_bal, cv=kf, scoring='accuracy')
logit_score = cross_val_score(logreg_pipe, features_bal, labels_bal, cv=kf, scoring='accuracy')

# Print the mean of each array o scores
print("Decision Tree:", np.mean(tree_score), "Logistic Regression:", np.mean(logit_score))
# final model: logistic regression









