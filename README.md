# CLFS-DS-Proteins
Python scripts to analyze a set of 77 proteins taken for wild-type and Down syndrome mice that have undergone stimulation and treatment. Goal is to extract proteins important to discriminate between the various classes. Analysis also performed of how different classifiers affect this problem.

Last updated May 4th, 2018

## Data Pre-Analysis
#### Files 
1. missingdata.py

#### Function
1. Views missing samples/features in the dataset

## Preprocessing, Decision Tree and Random Forest Feature Selection
#### Files
1. DT_feature_selection.py
2. RF_feature_selection.py

#### Function
1. Preprocessing of the data is performed
2. Grid search for optimized parameters for both Decision Tree and Random Forest
3. Training and Testing of Decision Tree and Random Forest
4. Perform RFECV of Decision Tree and Random Forest
5. Training and Testing of Decision Tree and Random Forest, with Feature Selection

## Neural Network Feature Selection
#### Files
1. Project_ANN_Complete.py
2. Project_ANN_Hidden_Layer_Analysis.py

#### Function
1. Preprocessing of the data is performed
2. The best number of hidden layer nodes for a given number of features is estimated
3. Perform RFECV of ANN
4. Grid search for optimized hidden layer nodes for Whole-ANN (all features), Sub-ANN (DT features), For-ANN (RF features), and RFE-ANN
5. Training and Testing of each of the neural networks

## Dimensionality Reduction and Unsupervised Clustering
#### Files
1. Project_Clustering.py
2. Project_Clustering_Averages.py
3. Project_KMeans_RFECV.py

#### Function
1. Preprocessing of the data is performed
2. 3 component PCA and LDA performed for full set of features, and features averaged per mouse
3. K-means clustering of full set of features, and features averaged per mouse
4. Perform RFECV of K-Means

## Feature Analysis
#### Files
1. Project_Pubmed_Miner.py
2. Project_VennDiagrams.py

#### Function
1. Query PubMed for literature results and show results as pie charts
2. Produces Venn diagrams showing feature selection overlaps between the various methods

