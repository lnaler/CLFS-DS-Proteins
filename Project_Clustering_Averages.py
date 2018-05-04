
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 02:19:54 2018

@author: Lynette
"""
#Peforms clustering on the samples averaged per mouse: PCA, LDA, K-Means

#IMPORTS
import os
import pandas
import numpy

from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn import metrics 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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

min_max_scaler = preprocessing.MinMaxScaler()
features = min_max_scaler.fit_transform(features)

#This makes sure to average the proteins per mouse
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
    
features_avg = numpy.array(averages)
labels_avg = numpy.array(avg_labels)
named_labels = numpy.array(label_names)

###############################################################################
#PCA - Unsupervised Clustering
pca = PCA(n_components = 3)
pca_feats = pca.fit_transform(features_avg)

fig = plt.figure(figsize=(8, 6))
ax = Axes3D(fig)
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'sienna']

for color, i, name in zip(colors, [0, 1, 2, 3, 4, 5, 6, 7], set(named_labels)):
    ax.scatter(pca_feats[labels_avg == i,0],
               pca_feats[labels_avg == i,1],
               pca_feats[labels_avg == i,2],
               color = color,
               label = name,
               s = 40)
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])
ax.set_title("PCA")
ax.legend()
plt.show()

###############################################################################
#LDA - Supervised Clustering
fig = plt.figure(figsize=(8, 6))
ax = Axes3D(fig)
lda = LinearDiscriminantAnalysis(n_components=3)
lda_feats = lda.fit(features_avg, labels_avg).transform(features_avg)
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'sienna']

for color, i, name in zip(colors, [0, 1, 2, 3, 4, 5, 6, 7], set(named_labels)):
    ax.scatter(lda_feats[labels_avg == i,0],
               lda_feats[labels_avg == i,1],
               lda_feats[labels_avg == i,2],
               color = color,
               label = name,
               s=40)
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])
ax.set_title("LDA")
ax.legend()
plt.show()

###############################################################################
#K Means
#Dataset - Complete Averaged Dataset

kmeans = KMeans(n_clusters=8,n_init=10)
kmeans = kmeans.fit(features_avg)
sil_score = metrics.silhouette_score(features_avg, labels_avg, metric='euclidean')
total_score = metrics.homogeneity_completeness_v_measure(labels_avg, kmeans.labels_)
print("K means silhouette score: ", sil_score)
print("K means homogeneity score: ", total_score[0])
print("K means completeness score: ", total_score[1])
print("K means v measure score: ", total_score[2])