# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 21:36:14 2018

@author: Lynette
"""

#IMPORTS
import os
import pandas
import numpy

from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn import metrics 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

#Setup confusion matrix function
def K_Means_RFE(feature_set, label_set, depth_index, score_spread):
    """A recursive function to extract the best features and score vs # of features"""
    if len(feature_set[0]) == 1:
        return (-2,-2,-2), [], score_spread
    best_features = []
    max_sil_score = (-2,-2,-2) #since range is 0 to 1, this will be overridden
    top_features = []
    for i in range(len(feature_set[0])):
        sub_set = numpy.delete(feature_set,i,1)
        kmeans = KMeans(n_clusters=8,n_init=10)
        kmeans = kmeans.fit(sub_set)
        sil_score = metrics.homogeneity_completeness_v_measure(label_set, kmeans.labels_)
        if sil_score[2] > max_sil_score[2]:
                max_sil_score = sil_score
                top_features = sub_set
    score_spread = numpy.insert(score_spread, 0, max_sil_score[2])
    
    print("Now entering depth: ", depth_index+1)
    best_score, best_features, score_spread = K_Means_RFE(top_features, label_set, depth_index+1, score_spread)
    if max_sil_score[2] > best_score[2]:
            best_score = max_sil_score
            best_features = top_features
    print("Now leaving depth: ", depth_index)
    return best_score, best_features, score_spread

#Import the training and testing datasets
dir_name = os.path.abspath(os.path.dirname(__file__))

data_file = 'processed_data.csv'
file_path = os.path.join(dir_name, data_file)
data = pandas.read_csv(file_path)
named_labels = data.iloc[:,-1]
encoder = preprocessing.LabelEncoder() 
encoder.fit(data.iloc[:,-1])
data.iloc[:,-1] = encoder.transform(data.iloc[:,-1])

#Prep the features and labels
features = numpy.array(data.iloc[:,1:78])
labels = numpy.array(data.iloc[:,-1])

###############################################################################
#K Means - ALL
###############################################################################
best_score, best_features, score_spread = K_Means_RFE(features, labels, 0, numpy.array([]))
print("Best score found: ", best_score[2], " and number of features: ", len(best_features[0]))
print("Homogeneity: ", best_score[0])
print("Completeness: ", best_score[1])

k = range(len(features[0])-1)
plt.figure()
linewidth = 2 
plt.plot(k, score_spread, lw = linewidth)
plt.xlim([0.0, 76])
plt.ylim([-2, 2])
plt.xlabel('Number of Features Kept')
plt.ylabel('V Measure')
plt.title('K Means Recursive Feature Elimination')
plt.legend(loc = "lower right")
plt.show()

###############################################################################
#K Means - AVERAGES
###############################################################################
averages = []
avg_labels = []
label_names = []
s = data.iloc[0, 0]
curr_mouse = s.split("_")[0]
min_idx = 0
for i in range(len(features)):
    s = data.iloc[i, 0]
    mouse = s.split("_")[0]
    if mouse != curr_mouse:
        max_idx = i
        current_row = features[min_idx:max_idx,:].mean(axis=0)
        avg_labels.append(labels[min_idx])
        averages.append(current_row)
        label_names.append(named_labels[min_idx])
        min_idx = i
        curr_mouse = mouse
max_idx = i
current_row = features[min_idx:max_idx,:].mean(axis=0)
avg_labels.append(labels[min_idx])
averages.append(current_row)
label_names.append(named_labels[min_idx])
   
features = numpy.array(averages)
labels = numpy.array(avg_labels)
best_score, best_features_avg, score_spread = K_Means_RFE(features, labels, 0, numpy.array([]))
print("Best score found: ", best_score[2], " and number of features: ", len(best_features_avg[0]))
print("Homogeneity: ", best_score[0])
print("Completeness: ", best_score[1])


k = range(len(features[0])-1)
plt.figure()
linewidth = 2 
plt.plot(k, score_spread, lw = linewidth)
plt.xlim([1, 77])
plt.ylim([-2, 2])
plt.xlabel('Number of Features Kept')
plt.ylabel('V Measure')
plt.title('Average K Means Recursive Feature Elimination')
plt.legend(loc = "lower right")
plt.show()

###############################################################################

numpy.savetxt("avg_kmeans_feats.csv", best_features_avg, delimiter=",")
numpy.savetxt("all_kmeans_feats.csv", best_features, delimiter=",")