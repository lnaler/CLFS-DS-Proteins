# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 00:54:29 2018

@author: Lynette
"""
#IMPORTS
import os
import pandas
import numpy
import itertools

from sklearn import preprocessing
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix
from matplotlib import cm

###############################################################################
#Setup confusion matrix function
def plot_confusion_matrix(cm, classes,
                          normalize = False,
                          title = 'Confusion matrix',
                          cmap = pyplot.cm.Blues):
    """This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    pyplot.figure()
    pyplot.imshow(cm, interpolation = 'nearest', cmap=cmap)
    pyplot.title(title)
    tick_marks = numpy.arange(len(classes))
    pyplot.xticks(tick_marks, classes, rotation = 45)
    pyplot.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, numpy.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    thresh = 2*cm.max() / 3.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        pyplot.text(j, i, "{:0.2f}".format(cm[i, j]), horizontalalignment = "center", 
                 color = "white" if cm[i, j] > thresh else "black")
    pyplot.tight_layout()
    pyplot.ylabel('True label')
    pyplot.xlabel('Predicted label')

###############################################################################
    
def ANN_RFECV(feature_set, label_set, test_set, score_spread):
    """A recursive function to extract the best features and score vs # of features"""
    if len(feature_set[0]) == 1:
        return 0, [], score_spread, []
    
    best_features = []
    len_feats = len(feature_set[0])
    worst_index = -1
    max_score = 0 #since range is 0 to 1, this will be overridden
    top_features = []
    print("Now processing: ", len_feats)
    
    for i in range(len_feats):
        layer_1 = int(round(len_feats*0.7,0))
        layer_2 = int(round(layer_1*0.8,0))
        sub_set = numpy.delete(feature_set,i,1)
        classifier = MLPClassifier(solver='lbfgs', hidden_layer_sizes = (layer_1, layer_2))
        scores = cross_val_score(classifier, sub_set, label_set, cv = 5)
        if scores.mean() > max_score:
                max_score = scores.mean()
                top_features = sub_set
                worst_index=i
    score_spread = numpy.insert(score_spread, 0, max_score)
    test_set = numpy.delete(test_set,worst_index,1)
    
    best_score, best_features, score_spread, test_feats = ANN_RFECV(top_features, label_set, test_set, score_spread)
    if max_score > best_score:
            best_score = max_score
            best_features = top_features
            test_feats = test_set
    return best_score, best_features, score_spread, test_feats

###############################################################################

def Grid_Search(features, labels, save_file, title):
    """An iterative function to find the best i,j values for hidden layer nodes"""
    #ANN - All features
    best_i = 0
    best_j = 0
    best_score = 0
    num_feats = len(features[0])
    
    cv_all = []
    cv = []
    for i in range (2,num_feats+1):
        cv = [] 
        for j in range (2,num_feats+1):
            print("Size: ", i, ", ", j)
            classifier = MLPClassifier(solver='lbfgs', hidden_layer_sizes = (i,j))
            scores = cross_val_score(classifier, features_train_val, labels_train_val, cv = 5)
            cv.append(scores.mean())
            if(scores.mean() > best_score):
                best_score = scores.mean()
                best_i = i
                best_j = j
        cv_all.append(cv)
    
    i = range(2,num_feats+1)
    j = range(2,num_feats+1)
    i, j = numpy.meshgrid(i, j)
    cv_all = numpy.transpose(numpy.array(cv_all))
    fig = pyplot.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(i, j, cv_all, cmap=cm.coolwarm)
    ax.set_xlabel('Nodes Layer 1')
    ax.set_ylabel('Nodes Layer 2')
    ax.set_zlabel('Mean CV Score')
    ax.set_title(title)
    pyplot.show()
    pyplot.savefig(save_file + '2.png')
    return best_i, best_j, best_score
    
###############################################################################

def runTesting(train_feats, test_feats, train_labels, test_labels, i, j, score, title):
    print(title)
    print("i: ", i, "j: ", j, "score: ", score)
    classifier = MLPClassifier(solver='lbfgs', hidden_layer_sizes = (i, j))
    classifier = classifier.fit(train_feats, train_labels)
    predicted_labels = classifier.predict(test_feats)
    
    conf_matrix = confusion_matrix(test_labels, predicted_labels)
    numpy.set_printoptions(precision = 2)
    plot_confusion_matrix(conf_matrix, 
                          classes = ("0","1","2","3","4","5","6","7"), 
                          title = title,
                          normalize = True)
    class_labels = ['0','1','2','3','4','5','6','7']
    print(metrics.classification_report(test_labels,predicted_labels,class_labels))

###############################################################################
#Import the training and testing datasets
dir_name = os.path.abspath(os.path.dirname(__file__))

sub_feat_cols = [0, 3, 5, 7, 10, 13, 17, 20, 30, 32, 33, 37, 39, 41, 42, 46, 48,
                 52, 53, 54, 55, 56, 60, 62, 63, 65, 66] #Decision Tree Features

for_feat_cols = [0, 1, 2, 7, 10, 11, 12, 13, 15, 17, 18, 19, 20, 21, 30, 32, 33,
                 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 48, 49, 50, 52,
                 53, 54, 55, 56, 58, 59, 60, 61, 62, 63, 64, 65, 67, 68, 69, 70,
                 72, 73, 75, 76] #Random Forest Features

data_file = 'processed_data.csv'
file_path = os.path.join(dir_name, data_file)

data = pandas.read_csv(file_path)
encoder = preprocessing.LabelEncoder() 
encoder.fit(data.iloc[:,-1])
data.iloc[:,-1] = encoder.transform(data.iloc[:,-1])

#Prep the features and labels
features = numpy.array(data.iloc[:,1:78])
labels = numpy.array(data.iloc[:,-1])

min_max_scaler = preprocessing.MinMaxScaler()
features = min_max_scaler.fit_transform(features)

#We want a 80:20 split
#train/crossval & test split
features_train_val, features_test, labels_train_val, labels_test = train_test_split(features, labels, test_size = 0.20)
sub_features_train_val = numpy.array(features_train_val[:,sub_feat_cols])
sub_features_test = numpy.array(features_test[:,sub_feat_cols])

for_features_train_val = numpy.array(features_train_val[:,for_feat_cols])
for_features_test = numpy.array(features_test[:,for_feat_cols])

whole_i, whole_j, whole_score = Grid_Search(features_train_val, labels_train_val, 'whole-ann-grid', 'Whole-ANN Grid Search')
sub_i, sub_j, sub_score = Grid_Search(sub_features_train_val, labels_train_val, 'sub-ann-grid', 'Sub-ANN Grid Search')
for_i, for_j, for_score = Grid_Search(for_features_train_val, labels_train_val, 'for-ann-grid', 'For-ANN Grid Search')

###############################################################################
#ANN RFECV Calc
best_score, best_features, score_spread, test_features = ANN_RFECV(features_train_val, labels_train_val, features_test, numpy.array([]))
print("Best score found: ", best_score, " and number of features: ", len(best_features[0]))
print("Number of test features: ", len(test_features[0]))

k = range(len(features[0])-1)
pyplot.figure()
linewidth = 2 
pyplot.plot(k, score_spread, lw = linewidth)
pyplot.xlim([0, 76])
pyplot.ylim([0, 1])
pyplot.xlabel('Number of Features Kept')
pyplot.ylabel('CV Score')
pyplot.title('Neural Network Recursive Feature Elimination')
pyplot.legend(loc = "lower right")
pyplot.show()

numpy.savetxt("ann_feats.csv", best_features, delimiter=",")
numpy.savetxt("ann_test_feats.csv", test_features, delimiter=",")

rfe_i, rfe_j, rfe_score = Grid_Search(best_features, labels_train_val, 'rfe-ann-grid', 'RFECV-ANN Grid Search')
rfe_features_train_val = best_features
rfe_features_test = test_features
###############################################################################
#Whole-ANN Testing
runTesting(features_train_val, features_test, labels_train_val, labels_test, whole_i, whole_j, whole_score, 'Whole-ANN')
#Sub-ANN Testing
runTesting(sub_features_train_val, sub_features_test, labels_train_val, labels_test, sub_i, sub_j, sub_score, 'Sub-ANN')
#Rfe-ANN Testing
runTesting(rfe_features_train_val, rfe_features_test, labels_train_val, labels_test, rfe_i, rfe_j, rfe_score, 'RFE-ANN')
#For-ANN Testing
runTesting(for_features_train_val, for_features_test, labels_train_val, labels_test, for_i, for_j, for_score, 'Forest-ANN')