#!/usr/bin/env python 

"""                                                                                                         
Created on Sat Apr 14 22:50:42 2018                                                                                                                                                                                                                                                                                                      
@author: Noushin                                                                                                                                                            
"""

import os
from time import time
import itertools
import numpy as np
import numpy
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from sklearn import preprocessing
from scipy.stats import randint as sp_randint

from sklearn.metrics import confusion_matrix
from matplotlib import cm

# Utility function to report best scores                                                                                                                                   
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

#Setup confusion matrix function   
def plot_confusion_matrix(cm, classes,
                          normalize = False,
                          title = 'Confusion matrix',
                          cmap = plt.cm.Blues):
    """This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.    """

    plt.imshow(cm, interpolation = 'nearest', cmap=cmap)
    plt.title(title)
    tick_marks = numpy.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, numpy.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    thresh = 2*cm.max() / 3.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:0.2f}".format(cm[i, j]), horizontalalignment = "center",
                 color = "white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

'***************************************** Loading Data*****************************************'

dir_name = os.path.abspath(os.path.dirname(__file__))
data_file = 'Data_Cortex_Nuclear.xls'
data_file_path = os.path.join(dir_name, data_file)
df0 = pd.read_excel(data_file_path,header=0)
df_isize= df0.shape

classes=["c-CS-m","c-SC-m","c-CS-s", "c-SC-s","t-CS-m","t-SC-m","t-CS-s","t-SC-s"]

names= list(df0.columns.values)[1:-4]
prot_names=[x.encode('UTF8')[:-2] for x in names]

'******************************************Preprocessing***************************************'

print 'number of missing values',df0.isnull().sum().sum()

# Excluding the samples which had missing values for the majority of proteins
df=df0[df0.isnull().sum(axis=1)<40]
df_size=df.shape
print df_isize[0]-df_size[0], 'of samples had more than 40 missing values. These mice were excluded'
print 'number of missing values',df.isnull().sum().sum()

mean_class={}
# Getting the similar class mean values of each feature  
for i in range(8):
    mean_class[i]= df[df['class'].str.contains(classes[i])].mean()

#replacing the missing values with the mean of similar class     
for c in range(8):
    for name in names :
        df.update(df.loc[df['class'].str.contains(classes[c])][name].fillna(mean_class[c][name]))

features = numpy.array(df.iloc[:, range(1,78)])
labels = numpy.array(df.iloc[:, -1])

# Multiple box plots on one Axe befor normalization for first 20 features
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.boxplot(features[:,range(0,20)],showfliers=True)
ax1.set_axisbelow(True)
ax1.set_title('Expression level of proteins')
ax1.set_xlabel('Proteins')
ax1.set_ylabel('Expression level')
ax1.set_xticklabels(np.repeat(prot_names[0:20], 2),
                    rotation=45, fontsize=8)
plt.savefig('before.png')

# normalization
min_max_scaler = preprocessing.MinMaxScaler()
features = min_max_scaler.fit_transform(features)

# Multiple box plots on one Axe after normalization for first 20 features 
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.boxplot(features[:,range(0,20)],showfliers=True)
ax2.set_axisbelow(True)
ax2.set_title('Expression level of proteins')
ax2.set_xlabel('Proteins')
ax2.set_ylabel('Expression level')
ax2.set_xticklabels(np.repeat(prot_names[0:20], 2),
                    rotation=45, fontsize=8)
plt.savefig('after.png')

# Encoding class labels
encoder = preprocessing.LabelEncoder( )
encoder.fit(labels)
labels = encoder.transform(labels)

# Data Splite
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = (train_test_split(features, labels, test_size = 0.2))

# Building classifier
from sklearn import tree
classifier = tree.DecisionTreeClassifier()


'************************* Grid search over Decision tree parameters*********************************'

from sklearn.model_selection import GridSearchCV

param_grid = {"max_depth": [3, None],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "criterion": ["gini", "entropy"]}

grid_search = GridSearchCV(classifier, param_grid=param_grid)

start = time()
grid_search.fit(features_test, labels_test)
print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)

classifier= grid_search.best_estimator_


'***********************Evaluation of the classifier without feature selection**************************'

fitted=classifier.fit(features_train,labels_train)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier, features,labels, cv = 10)

print('After cross validation, Scores are: ', scores * 100)
print('Mean: ', scores.mean(), ' and Standard Deviation: ', scores.std())

from sklearn.metrics import classification_report
predicted = fitted.predict(features_test)
print(classification_report(labels_test, predicted, target_names=classes))

conf_matrix = confusion_matrix(labels_test, predicted)
np.set_printoptions(precision = 2)
plt.figure('Figure 2')
plot_confusion_matrix(conf_matrix,
                      classes = ("0","1","2","3","4","5","6","7"),
                      title = 'Decision Tree without feature selection',
                      normalize = True)
plt.savefig('No_feature_selection.png',bbox_inches='tight')


'******************************************Feature Selection****************************************'

# Eliminate features recursively 
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
rfecv = RFECV(estimator = classifier, step = 1, cv = StratifiedKFold(10), scoring = 'accuracy')
rfecv.fit(features_train, labels_train)

features_train = rfecv.transform(features_train) # keeps only the remaining features after elemination                                                 
features_test = rfecv.transform(features_test)

print("Optimal number of features : %d" % rfecv.n_features_)
idx=[i for i, x in enumerate(rfecv.support_) if x]
print "Selected Features: %s" % df.columns.values[1:78][idx]
print 'accuracy of the model based on selected features is ', rfecv.grid_scores_[rfecv.n_features_]

'***********************Evaluation of the classifier with feature selection*************************'

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.savefig('CV_No_of_Features.pdf')

conf_matrix = confusion_matrix(labels_test, predicted)
np.set_printoptions(precision = 2)
plt.figure('Figure 1')
plot_confusion_matrix(conf_matrix,
                      classes = ("0","1","2","3","4","5","6","7"),
                      title = 'Decision Tree with feature selection',
                      normalize = True)
plt.savefig('Feature_selection.png',bbox_inches='tight')

from sklearn.metrics import classification_report
fitted2=rfecv.fit(features_train, labels_train)
predicted = fitted2.predict(features_test)
print(classification_report(labels_test, predicted, target_names=classes))
